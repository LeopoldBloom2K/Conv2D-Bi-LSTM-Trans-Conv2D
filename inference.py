import torch
import librosa
import soundfile as sf
import numpy as np
import argparse
import os
from models.crnn_separator import CRNN_Separator

def separate_audio(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"Loading model from {args.model_path}...")
    # 모델 초기화 (512 bin)
    model = CRNN_Separator(n_bins=512).to(device)
    state_dict = torch.load(args.model_path, map_location=device)
    
    if 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        model.load_state_dict(state_dict)
    
    model.eval()

    print(f"Processing audio: {args.input_path}")
    y, sr = librosa.load(args.input_path, sr=22050, mono=True)
    
    # 1. STFT 변환 (Shape: 513, Time)
    S = librosa.stft(y, n_fft=1024, hop_length=256)
    mag = np.abs(S)
    phase = np.angle(S)
    
    # [수정 1] 모델 입력용으로 513 -> 512로 자르기 (가장 높은 주파수 제거)
    mag_input = mag[:-1, :] 
    
    # 텐서 변환
    mag_tensor = torch.tensor(mag_input, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # 2. 추론
    with torch.no_grad():
        mask = model(mag_tensor)
        mask = mask.squeeze().cpu().numpy()

    # [옵션] 마스크 품질 향상 (Threshold & Scaling)
    mask = mask ** args.mask_scale
    mask[mask < args.threshold] = 0.0

    # 3. 마스크 적용 (Shape: 512, Time)
    est_mag_512 = mag_input * mask
    
    # [수정 2] iSTFT를 위해 512 -> 513으로 복구 (마지막 줄에 0 추가)
    # np.pad(배열, ((위, 아래), (왼, 오)))
    est_mag = np.pad(est_mag_512, ((0, 1), (0, 0)), mode='constant')

    # 4. 오디오 복원
    # 위상(phase)정보는 원본(513개)을 그대로 사용
    est_S = est_mag * np.exp(1j * phase)
    est_audio = librosa.istft(est_S, hop_length=256)

    # 저장
    filename = os.path.splitext(os.path.basename(args.input_path))[0]
    out_path = os.path.join(args.out_dir, f"{filename}_{args.target}_cleaned.wav")
    sf.write(out_path, est_audio, sr)
    
    print(f"✅ 완료! 저장됨: {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='./results')
    parser.add_argument('--target', type=str, default='vocals')
    parser.add_argument('--threshold', type=float, default=0.1)
    parser.add_argument('--mask_scale', type=float, default=2.0)
    
    args = parser.parse_args()
    separate_audio(args)