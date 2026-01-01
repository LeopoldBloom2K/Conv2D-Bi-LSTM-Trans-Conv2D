import torch
import numpy as np
import museval
import os
import sys
from tqdm import tqdm

# 프로젝트 루트 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.crnn_separator import CRNN_Separator
from utils.audio_processor import AudioProcessor

def find_optimal_gain():
    # 1. 체크포인트 규격에 맞게 강제 설정 (에러 해결 핵심)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SR = 22050      # 체크포인트 학습 사양
    N_FFT = 1024    # 체크포인트 학습 사양
    HOP_LENGTH = 256
    N_BINS = 512    # 8192 사이즈 불일치 해결을 위한 고정값
    HIDDEN_SIZE = 512
    NUM_LAYERS = 4
    
    # 파일 경로 설정 (사용자 경로)
    model_path = r"checkpoints\crnn_large_merged_0.9_0.1.pth" 
    test_dir = r"data\val"
    
    print(f"🚀 최적 Gain 탐색 시작 (구조 고정: n_bins=512, hidden=512)")
    
    # 2. 오디오 프로세서 및 모델 초기화
    processor = AudioProcessor(sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH)
    model = CRNN_Separator(
        input_channels=2, 
        n_bins=N_BINS, 
        num_stems=4,
        hidden_size=HIDDEN_SIZE, 
        num_layers=NUM_LAYERS
    ).to(device)

    # 가중치 로드
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    # 3. 데이터 로드
    song_folders = [f.path for f in os.scandir(test_dir) if f.is_dir()]
    if not song_folders:
        print("❌ 테스트 데이터를 찾을 수 없습니다.")
        return
        
    song_dir = song_folders[0]
    print(f"🎵 분석 대상: {os.path.basename(song_dir)}")
    
    mix_audio = processor.load_audio(os.path.join(song_dir, "mixture.wav"))
    ref_audio = processor.load_audio(os.path.join(song_dir, "vocals.wav"))
    
    min_len = min(mix_audio.shape[1], ref_audio.shape[1])
    mix_mag, mix_phase = processor.audio_to_stft(mix_audio[:, :min_len])
    
    with torch.no_grad():
        masks = model(mix_mag.unsqueeze(0).to(device))
        raw_mask = masks.squeeze(0).cpu().numpy()[0] 

    # 4. Gain 탐색 (0.5 ~ 8.0)
    gain_candidates = np.arange(0.5, 8.1, 0.2)
    best_sdr = -float('inf')
    best_gain = 1.0
    
    for gain in tqdm(gain_candidates, desc="SDR 최적화 중"):
        adjusted_mask = np.clip(raw_mask * gain, 0, 1)
        est_mag = mix_mag.numpy() * adjusted_mask
        est_audio = processor.stft_to_audio(est_mag, mix_phase)
        
        ref = ref_audio[:, :min_len].numpy().T[None, :, :]
        est = est_audio.T[None, :, :]
        
        sdr, _, _, _ = museval.evaluate(ref, est, win=min_len, hop=min_len)
        current_sdr = np.nanmedian(sdr)
        
        if current_sdr > best_sdr:
            best_sdr = current_sdr
            best_gain = gain

    print("\n" + "="*50)
    print(f"🏆 최적 Gain 결과: {best_gain:.1f}")
    print(f"📈 예상 최고 SDR: {best_sdr:.4f} dB")
    print("="*50)
    print(f"💡 이제 evaluate.py 실행 시 --gain {best_gain:.1f} 를 적용하세요.")

if __name__ == '__main__':
    find_optimal_gain()