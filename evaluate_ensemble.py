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

def evaluate_ensemble(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Ensemble Evaluation Device: {device}")
    
    SR = 22050
    N_FFT = 1024
    HOP_LENGTH = 256
    N_BINS = N_FFT // 2 
    
    # 1. 두 개의 모델 로드
    def load_weights(model, path, device):
        checkpoint = torch.load(path, map_location=device)
        # 키가 'model_state_dict' 내부에 있는지, 아니면 데이터 그 자체인지 확인
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        return model

    print(f"📦 Loading Model 1: {args.model_path1}")
    model1 = CRNN_Separator(input_channels=2, n_bins=N_BINS, num_stems=4, 
                            hidden_size=args.hidden_size, num_layers=args.num_layers).to(device)
    model1 = load_weights(model1, args.model_path1, device)

    print(f"📦 Loading Model 2: {args.model_path2}")
    model2 = CRNN_Separator(input_channels=2, n_bins=N_BINS, num_stems=4, 
                            hidden_size=args.hidden_size, num_layers=args.num_layers).to(device)
    model2 = load_weights(model2, args.model_path2, device)
    
    processor = AudioProcessor(sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH)
    
    # 2. 경로 확인
    test_path = os.path.abspath(args.test_dir)
    if os.path.isfile(test_path):
        test_path = os.path.dirname(test_path)
    
    if os.path.exists(os.path.join(test_path, "mixture.wav")):
        song_folders = [test_path]
    else:
        song_folders = [f.path for f in os.scandir(test_path) if f.is_dir()]
    
    print(f"🔍 총 {len(song_folders)}개의 곡을 앙상블 평가합니다.")
    
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

            mix_mag, mix_phase = processor.audio_to_stft(mix_audio)
            mix_mag_tensor = mix_mag.unsqueeze(0).to(device)

            with torch.no_grad():
                # 두 모델의 마스크를 각각 예측
                mask1 = model1(mix_mag_tensor)
                mask2 = model2(mix_mag_tensor)
                
                # 앙상블: 두 마스크의 산술 평균 (가장 안정적인 방식)
                # 가중치를 주고 싶다면 (mask1 * 0.4 + mask2 * 0.6) 식으로 조절 가능
                ensemble_mask = (mask1 + mask2) / 2.0
                
                # 옵션 적용 (앙상블 시에는 순정 1.0/0.1 추천)
                if args.mask_scale != 1.0:
                    ensemble_mask = ensemble_mask ** args.mask_scale
                
                masks_np = ensemble_mask.squeeze(0).cpu().numpy()

                if args.threshold > 0.0:
                    masks_np[masks_np < args.threshold] = 0.0

            stem_indices = {'vocals': 0, 'drums': 1, 'bass': 2, 'other': 3}
            mask = masks_np[stem_indices.get(args.target, 0)]

            # 복원 및 SDR 계산
            est_mag = mix_mag.cpu().numpy() * mask
            est_audio = processor.stft_to_audio(est_mag, mix_phase)

            ref = ref_audio.T[None, :, :]
            est = est_audio.T[None, :, :]
            L = min(ref.shape[1], est.shape[1])
            sdr, _, _, _ = museval.evaluate(ref[:, :L, :], est[:, :L, :], win=L, hop=L)
            
            sdr_results.append({'song': song_name, 'sdr': np.nanmedian(sdr)})
            
        except Exception as e:
            print(f"Error: {e}")
            continue

    df = pd.DataFrame(sdr_results)
    print("\n" + "="*40)
    print(f"🏆 앙상블 결과 ({args.target})")
    print(f"   - 평균 SDR: {df['sdr'].mean():.4f} dB")
    print("="*40)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, required=True)
    parser.add_argument('--target', type=str, default='vocals')
    # 두 개의 모델 경로를 받음
    parser.add_argument('--model_path1', type=str, required=True, help='Large V1 (3.79dB)')
    parser.add_argument('--model_path2', type=str, required=True, help='Final Polish (3.93dB)')
    
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--mask_scale', type=float, default=1.0)
    parser.add_argument('--threshold', type=float, default=0.1)
    
    args = parser.parse_args()
    evaluate_ensemble(args)