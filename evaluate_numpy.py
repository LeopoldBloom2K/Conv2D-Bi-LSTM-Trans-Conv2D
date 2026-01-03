import torch
import soundfile as sf
import numpy as np
import argparse
import os
import sys
from tqdm import tqdm
import pandas as pd

# 프로젝트 루트 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.crnn_separator import CRNN_Separator
from utils.audio_processor import AudioProcessor

def calculate_sdr_numpy(references, estimates):
    """
    Numpy를 이용한 고속 SDR 계산 함수 (BSSEval 대용)
    Input: (Samples, Channels) 형태의 Numpy 배열
    Output: SDR 값 (dB)
    """
    # 0으로 나누기 방지용 엡실론
    EPS = 1e-8
    
    # 1. 길이 맞추기 (안전장치)
    min_len = min(len(references), len(estimates))
    references = references[:min_len]
    estimates = estimates[:min_len]
    
    # 2. 오차 신호 (Noise) 계산
    delta = references - estimates
    
    # 3. 신호 강도(Power) 계산
    # Target Power: 정답 신호의 에너지
    s_target = np.sum(references ** 2) + EPS
    
    # Noise Power: 오차 신호의 에너지
    s_noise = np.sum(delta ** 2) + EPS
    
    # 4. SNR(SDR) 계산: 10 * log10(Signal / Noise)
    sdr = 10 * np.log10(s_target / s_noise)
    
    return sdr

def evaluate_dataset(args):
    # 1. 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 High-Speed Numpy Evaluation Device: {device}")
    
    SR = 22050
    N_FFT = 1024
    HOP_LENGTH = 256
    N_BINS = 512
    
    print(f"Loading model from {args.model_path}...")
    print(f"⚙️ Model Config: Hidden={args.hidden_size}, Layers={args.num_layers}")
    
    # 2. 모델 로드
    model = CRNN_Separator(
        input_channels=2, 
        n_bins=N_BINS, 
        num_stems=4,
        hidden_size=args.hidden_size,  
        num_layers=args.num_layers     
    ).to(device)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # 3. 오디오 프로세서
    processor = AudioProcessor(sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH)
    
    # 4. 곡 목록 확인
    test_path = os.path.abspath(args.test_dir)
    if os.path.isfile(test_path):
        test_path = os.path.dirname(test_path)

    if os.path.exists(os.path.join(test_path, "mixture.wav")):
        song_folders = [test_path]
    else:
        song_folders = [f.path for f in os.scandir(test_path) if f.is_dir()]
    
    print(f"🔍 총 {len(song_folders)}개의 곡을 평가합니다.")
    
    sdr_results = []
    
    for song_dir in tqdm(song_folders):
        try:
            song_name = os.path.basename(song_dir)
            mix_path = os.path.join(song_dir, "mixture.wav")
            target_path = os.path.join(song_dir, f"{args.target}.wav")
            
            if not os.path.exists(mix_path) or not os.path.exists(target_path):
                continue

            mix_audio = processor.load_audio(mix_path)
            ref_audio = processor.load_audio(target_path)
            
            min_len = min(mix_audio.shape[1], ref_audio.shape[1])
            mix_audio = mix_audio[:, :min_len]
            ref_audio = ref_audio[:, :min_len]

            # STFT
            mix_mag, mix_phase = processor.audio_to_stft(mix_audio)
            mix_mag_tensor = mix_mag.unsqueeze(0).to(device)

            with torch.no_grad():
                masks = model(mix_mag_tensor)
                
                # [옵션 1] Mask Scale
                if args.mask_scale != 1.0:
                    masks = masks ** args.mask_scale
                
                masks = masks.squeeze(0).cpu().numpy()

                # [옵션 2] Threshold
                if args.threshold > 0.0:
                    masks[masks < args.threshold] = 0.0
                
                stem_indices = {'vocals': 0, 'drums': 1, 'bass': 2, 'other': 3}
                mask = masks[stem_indices.get(args.target, 0)]

                # [옵션 3] Gain Correction
                if args.gain != 1.0:
                    mask = mask * args.gain
                    mask = np.clip(mask, 0, 1)

            # 복원
            est_mag = mix_mag.cpu().numpy() * mask
            est_audio = processor.stft_to_audio(est_mag, mix_phase)

            # --- SDR 계산 (Numpy 버전) ---
            # AudioProcessor는 (2, Samples)를 반환하므로 Transpose 필요
            ref_np = ref_audio.numpy().T  # (Samples, 2)
            est_np = est_audio.T          # (Samples, 2)
            
            score = calculate_sdr_numpy(ref_np, est_np)
            sdr_results.append({'song': song_name, 'sdr': score})
            
        except Exception as e:
            print(f"Error evaluating {song_name}: {e}")
            continue

    if not sdr_results:
        print("평가된 곡이 없습니다.")
        return

    df = pd.DataFrame(sdr_results)
    print("\n" + "="*40)
    print(f"📊 최종 평가 완료 (Numpy Fast Mode): {args.target}")
    print(f"   - Mask Scale: {args.mask_scale}")
    print(f"   - Threshold: {args.threshold}")
    print(f"   - Gain Correction: x{args.gain}")
    print(f"   - 평균 SDR: {df['sdr'].mean():.4f} dB")
    print("="*40)
    
    # 결과 파일 저장
    df.to_csv(f"eval_results_numpy_{args.target}.csv", index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, required=True)
    parser.add_argument('--target', type=str, default='vocals')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--mask_scale', type=float, default=1.0)
    parser.add_argument('--threshold', type=float, default=0.1)
    parser.add_argument('--gain', type=float, default=1.0, help='Linear gain multiplier')
    
    args = parser.parse_args()
    evaluate_dataset(args)