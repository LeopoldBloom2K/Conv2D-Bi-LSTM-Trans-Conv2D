import torch
import soundfile as sf
import argparse
from models.crnn_separator import CRNN_Separator
from utils.audio_processor import AudioProcessor

def separate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 모델 로드
    model = CRNN_Separator().to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    # 체크포인트 저장 방식에 따라 처리 (state_dict만 있거나 전체 dict이거나)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    processor = AudioProcessor(n_fft=2048, hop_length=512)
    
    # 2. 오디오 로드 및 전처리
    print(f"Processing: {args.input_path}")
    audio = processor.load_audio(args.input_path)
    mix_mag, mix_phase = processor.audio_to_stft(audio) # Phase 저장 필수!
    
    mix_mag_tensor = mix_mag.unsqueeze(0).to(device) # (1, 1, F, T)
    
    # 3. 추론
    with torch.no_grad():
        mask = model(mix_mag_tensor)
        predicted_mag = mix_mag_tensor * mask
        
    # 4. 복원 및 저장
    # 원본의 Phase 정보와 예측된 Magnitude를 합침
    estimated_audio = processor.stft_to_audio(predicted_mag, mix_phase)
    
    sf.write(args.output_path, estimated_audio, processor.sr)
    print(f"Separated audio saved to: {args.output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    
    args = parser.parse_args()
    separate(args)