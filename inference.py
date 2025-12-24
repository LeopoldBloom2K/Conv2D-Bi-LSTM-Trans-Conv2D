import torch
import numpy as np
import argparse
import os
import soundfile as sf
import sys

# 프로젝트 루트에서 실행 시 utils를 찾을 수 있도록 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.crnn_separator import CRNN_Separator
# [핵심] AudioProcessor를 utils 폴더에서 불러옵니다.
from utils.audio_processor import AudioProcessor

def separate_audio(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)
    
    # ====================================================
    # 1. 설정 (학습했던 options.py 설정과 똑같이 맞춰야 함)
    # ====================================================
    # options.py 기본값: sr=22050, n_fft=1024, hop_length=256
    SR = 22050
    N_FFT = 1024
    HOP_LENGTH = 256
    N_BINS = N_FFT // 2  # 512
    
    print(f"Loading model from {args.model_path}...")
    
    # 모델 초기화 (Stereo=2ch, 4 Stems)
    # audio_processor.py는 n_fft//2 (512)를 사용하므로 n_bins=N_BINS로 설정합니다.
    model = CRNN_Separator(input_channels=2, n_bins=N_BINS, num_stems=4).to(device)
    
    # 가중치 로드
    state_dict = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        model.load_state_dict(state_dict)
    
    model.eval()

    # ====================================================
    # 2. 오디오 로드 및 전처리 (AudioProcessor 사용)
    # ====================================================
    print(f"Processing audio: {args.input_path}")
    
    # processor 초기화 (설정값 전달 필수!)
    processor = AudioProcessor(sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH)
    
    # 오디오 로드 (자동으로 스테레오 변환됨) -> (2, Samples)
    mix_audio = processor.load_audio(args.input_path)
    
    # STFT 변환 -> Mag: (2, Freq, Time), Phase: (2, Freq, Time)
    # processor 내부에서 마지막 주파수 빈을 잘라 512개로 맞춰줍니다.
    mag, phase = processor.audio_to_stft(mix_audio)
    
    # 배치 차원 추가: (1, 2, Freq, Time)
    mag_tensor = mag.unsqueeze(0).to(device)

    # ====================================================
    # 3. 추론 (Inference)
    # ====================================================
    with torch.no_grad():
        # 결과: (1, 4, 2, Freq, Time) -> (Batch, Stems, Channels, Freq, Time)
        masks = model(mag_tensor)
        
        # 마스크 스케일링 (선택 사항, 분리도 조절)
        masks = masks ** args.mask_scale
        
        # 배치 차원 제거 -> (4, 2, Freq, Time)
        masks = masks.squeeze(0).cpu().numpy()
        
    # Threshold 적용 (너무 작은 값은 노이즈로 간주하고 제거)
    masks[masks < args.threshold] = 0.0

    # ====================================================
    # 4. 결과 복원 및 저장
    # ====================================================
    mag_np = mag.cpu().numpy() # (2, Freq, Time)
    
    stem_names = ['vocals', 'drums', 'bass', 'other']
    filename = os.path.splitext(os.path.basename(args.input_path))[0]
    
    print("Saving results...")
    
    # 4개 악기 트랙을 모두 저장합니다.
    for i, stem_name in enumerate(stem_names):
        # 해당 악기의 마스크: (2, Freq, Time)
        stem_mask = masks[i]
        
        # 마스킹: 원래 크기(Magnitude) * 마스크(Mask)
        est_mag = mag_np * stem_mask
        
        # 오디오 복원 (iSTFT) - 위상(Phase)은 원본 믹스의 것을 사용
        est_audio = processor.stft_to_audio(est_mag, phase)
        
        # soundfile 저장을 위해 (Channels, Samples) -> (Samples, Channels) 전치
        est_audio = est_audio.T 
        
        # 저장
        out_path = os.path.join(args.out_dir, f"{filename}_{stem_name}.wav")
        sf.write(out_path, est_audio, SR)
        print(f"  -> Saved: {out_path}")

    print("✅ 모든 분리 작업 완료!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help='분리할 노래 파일 경로')
    parser.add_argument('--model_path', type=str, required=True, help='학습된 모델(.pth) 경로')
    parser.add_argument('--out_dir', type=str, default='./results', help='결과 저장 폴더')
    parser.add_argument('--threshold', type=float, default=0.1, help='마스크 임계값 (노이즈 제거용)')
    parser.add_argument('--mask_scale', type=float, default=1.0, help='마스크 스케일 (높으면 더 강하게 분리)')
    
    args = parser.parse_args()
    separate_audio(args)