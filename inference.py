import torch
import numpy as np
import argparse
import os
import soundfile as sf
import sys

# 프로젝트 루트 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.crnn_separator import CRNN_Separator
from utils.audio_processor import AudioProcessor

def separate_audio(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 설정
    SR = 22050
    N_FFT = 1024
    HOP_LENGTH = 256
    N_BINS = N_FFT // 2 
    
    print(f"Loading model from {args.model_path}...")
    print(f"Model Config: Hidden={args.hidden_size}, Layers={args.num_layers}")
    
    # [수정] 옵션으로 받은 크기대로 모델 초기화
    model = CRNN_Separator(
        input_channels=2, 
        n_bins=N_BINS, 
        num_stems=4,
        hidden_size=args.hidden_size,  # 추가됨
        num_layers=args.num_layers     # 추가됨
    ).to(device)
    
    # 가중치 로드
    state_dict = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        model.load_state_dict(state_dict)
    
    model.eval()

    # 오디오 처리
    print(f"Processing audio: {args.input_path}")
    processor = AudioProcessor(sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mix_audio = processor.load_audio(args.input_path)
    
    mag, phase = processor.audio_to_stft(mix_audio)
    mag_tensor = mag.unsqueeze(0).to(device)

    # 추론
    with torch.no_grad():
        masks = model(mag_tensor)
        
        # 옵션 적용
        if args.mask_scale != 1.0:
            masks = masks ** args.mask_scale
        
        masks = masks.squeeze(0).cpu().numpy()
        
    if args.threshold > 0.0:
        masks[masks < args.threshold] = 0.0

    # 저장
    mag_np = mag.cpu().numpy()
    stem_names = ['vocals', 'drums', 'bass', 'other']
    filename = os.path.splitext(os.path.basename(args.input_path))[0]
    
    print("Saving results...")
    for i, stem_name in enumerate(stem_names):
        stem_mask = masks[i]
        est_mag = mag_np * stem_mask
        est_audio = processor.stft_to_audio(est_mag, phase)
        est_audio = est_audio.T 
        
        out_path = os.path.join(args.out_dir, f"{filename}_{stem_name}.wav")
        sf.write(out_path, est_audio, SR)
        print(f"  -> Saved: {out_path}")

    print("✅ 모든 분리 작업 완료!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help='Input audio path')
    parser.add_argument('--model_path', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--out_dir', type=str, default='./results', help='Output directory')
    
    # [추가됨] Large 모델 대응을 위한 옵션
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden size (256/512)')
    parser.add_argument('--num_layers', type=int, default=3, help='Num layers (3/4)')
    
    parser.add_argument('--threshold', type=float, default=0.1, help='Noise threshold')
    parser.add_argument('--mask_scale', type=float, default=1.0, help='Mask scaling factor')
    
    args = parser.parse_args()
    separate_audio(args)