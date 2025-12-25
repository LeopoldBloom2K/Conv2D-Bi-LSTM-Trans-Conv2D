import torch
import soundfile as sf
import numpy as np
import argparse
import os
import sys
import museval
from tqdm import tqdm
import pandas as pd

# 프로젝트 루트 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.crnn_separator import CRNN_Separator
from utils.audio_processor import AudioProcessor

def evaluate_dataset(args):
    # 1. 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluation Device: {device}")
    
    SR = 22050
    N_FFT = 1024
    HOP_LENGTH = 256
    N_BINS = N_FFT // 2  # 512
    
    # 2. 모델 로드
    model = CRNN_Separator(input_channels=2, n_bins=N_BINS, num_stems=4).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # 3. 오디오 프로세서
    processor = AudioProcessor(sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH)
    
    # 4. 곡 목록 확인
    if os.path.exists(os.path.join(args.test_dir, "mixture.wav")):
        song_folders = [args.test_dir]
    else:
        song_folders = [f.path for f in os.scandir(args.test_dir) if f.is_dir()]
    
    print(f"총 {len(song_folders)}개의 곡을 평가합니다.")
    print(f"옵션 적용: Mask Scale={args.mask_scale}, Threshold={args.threshold}")
    
    sdr_results = []
    
    for song_dir in tqdm(song_folders):
        try:
            song_name = os.path.basename(song_dir)
            mix_path = os.path.join(song_dir, "mixture.wav")
            target_path = os.path.join(song_dir, f"{args.target}.wav")
            
            if not os.path.exists(mix_path) or not os.path.exists(target_path):
                continue

            # 오디오 로드
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
                
                # --- [전략 2 핵심] 마스크 품질 향상 옵션 적용 ---
                # 1. Mask Scaling (분리도 강화)
                if args.mask_scale != 1.0:
                    masks = masks ** args.mask_scale
                
                masks = masks.squeeze(0).cpu().numpy()

                # 2. Thresholding (노이즈 제거)
                if args.threshold > 0.0:
                    masks[masks < args.threshold] = 0.0
                # ---------------------------------------------

            stem_indices = {'vocals': 0, 'drums': 1, 'bass': 2, 'other': 3}
            target_idx = stem_indices.get(args.target, 0)
            mask = masks[target_idx]

            # 복원
            est_mag = mix_mag.cpu().numpy() * mask
            est_audio = processor.stft_to_audio(est_mag, mix_phase)

            # SDR 계산
            ref = ref_audio.T[None, :, :]
            est = est_audio.T[None, :, :]
            
            L = min(ref.shape[1], est.shape[1])
            sdr, _, _, _ = museval.evaluate(ref[:, :L, :], est[:, :L, :], win=L, hop=L)
            
            score = np.nanmedian(sdr)
            sdr_results.append({'song': song_name, 'sdr': score})
            
        except Exception as e:
            print(f"Error evaluating {song_name}: {e}")
            continue

    if not sdr_results:
        print("평가된 곡이 없습니다.")
        return

    df = pd.DataFrame(sdr_results)
    print("\n" + "="*40)
    print(f"📊 평가 완료: {args.target}")
    print(f"   - Mask Scale: {args.mask_scale}")
    print(f"   - Threshold: {args.threshold}")
    print(f"   - 평균 SDR: {df['sdr'].mean():.4f} dB")
    print("="*40)
    
    df.to_csv(f"eval_results_{args.target}.csv", index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, required=True)
    parser.add_argument('--target', type=str, default='vocals')
    parser.add_argument('--model_path', type=str, required=True)
    
    # [추가됨] 전략 2를 위한 옵션
    parser.add_argument('--mask_scale', type=float, default=1.0, help='마스크 선명도 조절 (보통 1.2~1.5 추천)')
    parser.add_argument('--threshold', type=float, default=0.1, help='노이즈 제거 임계값 (보통 0.1~0.2 추천)')
    
    args = parser.parse_args()
    evaluate_dataset(args)