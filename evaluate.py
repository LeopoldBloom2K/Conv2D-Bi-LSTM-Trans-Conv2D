import torch
import soundfile as sf
import numpy as np
import argparse
import os
import museval
from torch.amp import autocast
from models.crnn_separator import CRNN_Separator
from utils.audio_processor import AudioProcessor

def evaluate_song(args):
    # 1. 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluation Device: {device}")
    
    # 2. 모델 로드 (n_bins=512 필수)
    model = CRNN_Separator(n_bins=512).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # 3. 오디오 프로세서
    processor = AudioProcessor(sr=22050, n_fft=1024, hop_length=256)
    
    # 4. 파일 경로 확인
    # args.song_dir 예: "D:\musdb18hq\test\Signe - My Song"
    mix_path = os.path.join(args.song_dir, "mixture.wav")
    target_path = os.path.join(args.song_dir, f"{args.target}.wav")
    
    if not os.path.exists(mix_path) or not os.path.exists(target_path):
        print("❌ 파일이 없습니다. 경로를 확인하세요.")
        return

    # 5. 오디오 로드
    print(f"Loading: {args.song_dir}")
    mix_audio = processor.load_audio(mix_path)
    ref_audio = processor.load_audio(target_path) # 정답(Reference)
    
    # --- 추론 (Chunking 없이 통으로 하거나, 메모리 부족시 Chunking 적용 필요) ---
    # 평가의 정확도를 위해 여기서는 통으로 처리하되, 메모리 관리 주의
    # (긴 곡은 inference.py의 청크 로직을 가져와야 함. 여기선 30초만 잘라서 테스트 추천)
    
    # 테스트를 위해 앞부분 30초만 잘라서 평가 (속도 UP)
    # 전체 곡 평가를 원하면 이 부분 슬라이싱([: ...])을 제거하세요.
    test_len = min(len(mix_audio), 30 * processor.sr) 
    mix_audio = mix_audio[:test_len]
    ref_audio = ref_audio[:test_len]

    # STFT
    mix_mag, mix_phase = processor.audio_to_stft(mix_audio)
    mix_mag_tensor = mix_mag.unsqueeze(0).to(device)

    with torch.no_grad():
        with autocast(device_type='cuda'):
            mask = model(mix_mag_tensor)
            pred_mag = mix_mag_tensor * mask
            
    # 복원 (Estimate)
    est_audio = processor.stft_to_audio(pred_mag, mix_phase)
    
    # 6. 길이 맞추기 (museval은 길이가 1샘플이라도 다르면 에러남)
    min_len = min(len(ref_audio), len(est_audio))
    ref_audio = ref_audio[:min_len]
    est_audio = est_audio[:min_len]
    
    # 7. SDR 계산
    print("Calculating SDR Score...")
    
    # museval 입력 형태: (nsrc, samples, channels)
    # 우리는 Mono이므로 (1, samples, 1)로 변환
    references = ref_audio[None, :, None] 
    estimates = est_audio[None, :, None] 
    
    # win=샘플수 (곡 전체를 하나의 윈도우로 평가)
    sdr, isr, sir, sar = museval.evaluate(references, estimates, win=min_len, hop=min_len)
    
    sdr_score = np.nanmedian(sdr)
    print("------------------------------------------------")
    print(f"🎵 Target Instrument: {args.target}")
    print(f"📈 SDR Score: {sdr_score:.4f} dB")
    print("------------------------------------------------")
    
    # 8. 결과 저장 (들어보기 위해)
    out_path = "eval_result.wav"
    sf.write(out_path, est_audio, processor.sr)
    print(f"🔊 분리된 파일 저장됨: {out_path} (들어보세요!)")

    return sdr_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--song_dir', type=str, required=True, help='Path to a test song folder')
    parser.add_argument('--target', type=str, default='vocals')
    parser.add_argument('--model_path', type=str, default='./checkpoints/best_model.pth')
    
    args = parser.parse_args()
    evaluate_song(args)