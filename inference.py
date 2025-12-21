import torch
import soundfile as sf
import argparse
import os
import numpy as np
from tqdm import tqdm
from torch.amp import autocast 
from models.crnn_separator import CRNN_Separator
from utils.audio_processor import AudioProcessor

def separate(args):
    # 1. 디바이스 설정
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f"Device: {device}")

    # 2. 모델 로드 (n_bins=512 필수)
    model = CRNN_Separator(n_bins=512).to(device)
    
    if not os.path.exists(args.model_path):
        print(f"Error: 모델 파일이 없습니다 -> {args.model_path}")
        return

    print(f"Loading model: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    
    # 3. 오디오 프로세서 (학습과 동일 설정)
    processor = AudioProcessor(sr=22050, n_fft=1024, hop_length=256)
    
    # 4. 오디오 로드
    print(f"Loading audio: {args.input_path}")
    full_audio = processor.load_audio(args.input_path)
    
    # --- [핵심] 청크(Chunk) 단위 처리 ---
    # 노래를 통째로 넣으면 느리니까, 10초씩 잘라서 처리합니다.
    chunk_seconds = 10 
    chunk_samples = chunk_seconds * processor.sr
    total_len = len(full_audio)
    
    # 결과물을 담을 빈 리스트
    output_audio = np.zeros_like(full_audio)
    
    print(f"Start separating (Total length: {total_len/processor.sr:.1f}s)...")
    
    # 진행바 표시
    for i in tqdm(range(0, total_len, chunk_samples)):
        # 10초씩 자르기
        chunk = full_audio[i : i + chunk_samples]
        
        # 마지막 조각이 너무 짧으면 패딩
        pad_len = 0
        if len(chunk) < chunk_samples:
            pad_len = chunk_samples - len(chunk)
            chunk = np.pad(chunk, (0, pad_len))
            
        # 전처리
        mix_mag, mix_phase = processor.audio_to_stft(chunk)
        mix_mag_tensor = mix_mag.unsqueeze(0).to(device)
        
        # 추론
        with torch.no_grad():
            with autocast(device_type='cuda', enabled=use_cuda):
                mask = model(mix_mag_tensor)
                pred_mag = mix_mag_tensor * mask
        
        # 복원
        estimated_chunk = processor.stft_to_audio(pred_mag, mix_phase)
        
        # 패딩했던 부분 다시 잘라내기
        if pad_len > 0:
            estimated_chunk = estimated_chunk[:-pad_len]
            
        # 결과 저장
        # (주의: STFT 복원 과정에서 길이가 미세하게 달라질 수 있어 길이 맞춤)
        write_len = min(len(estimated_chunk), len(output_audio[i : i + chunk_samples]))
        output_audio[i : i + write_len] = estimated_chunk[:write_len]

    # 5. 저장
    os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
    sf.write(args.output_path, output_audio, processor.sr)
    print(f"\n✨ 완료! 저장됨: {args.output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default='result.wav')
    parser.add_argument('--model_path', type=str, default='./checkpoints/best_model.pth')
    
    args = parser.parse_args()
    separate(args)