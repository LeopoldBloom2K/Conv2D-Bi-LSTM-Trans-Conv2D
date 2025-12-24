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
    # 1. 설정 (학습 모델과 동일하게 고정)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluation Device: {device}")
    
    SR = 22050
    N_FFT = 1024        # 학습 설정
    HOP_LENGTH = 256    # 학습 설정
    N_BINS = 512        # n_fft // 2
    
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
    
    # 4. 평가할 곡 리스트업
    # 입력된 폴더가 노래 폴더인지, 노래들이 들어있는 상위 폴더인지 확인
    if os.path.exists(os.path.join(args.test_dir, "mixture.wav")):
        # 단일 곡 폴더인 경우
        song_folders = [args.test_dir]
    else:
        # 상위 폴더인 경우 (모든 하위 폴더 검색)
        song_folders = [f.path for f in os.scandir(args.test_dir) if f.is_dir()]
    
    print(f"총 {len(song_folders)}개의 곡을 평가합니다.")
    
    # 결과 저장용 리스트
    sdr_results = []
    
    # 5. 반복 평가 시작
    for song_dir in tqdm(song_folders):
        try:
            song_name = os.path.basename(song_dir)
            mix_path = os.path.join(song_dir, "mixture.wav")
            target_path = os.path.join(song_dir, f"{args.target}.wav")
            
            if not os.path.exists(mix_path) or not os.path.exists(target_path):
                # 파일이 없으면 스킵
                continue

            # 오디오 로드 (Stereo)
            mix_audio = processor.load_audio(mix_path)
            ref_audio = processor.load_audio(target_path)
            
            # --- 길이 맞추기 (30초 제한 해제 가능) ---
            # 전체 평가를 위해 30초 제한을 풀거나, 속도를 위해 유지할 수 있습니다.
            # 여기선 정확도를 위해 전체 길이를 사용하되, 메모리 부족 시 조절하세요.
            min_len = min(mix_audio.shape[1], ref_audio.shape[1])
            
            # 너무 긴 곡은 메모리 터질 수 있으니 최대 1분(60초)까지만 평가 (옵션)
            # max_samples = 60 * SR
            # min_len = min(min_len, max_samples)

            mix_audio = mix_audio[:, :min_len]
            ref_audio = ref_audio[:, :min_len]

            # STFT 및 추론
            mix_mag, mix_phase = processor.audio_to_stft(mix_audio)
            mix_mag_tensor = mix_mag.unsqueeze(0).to(device)

            with torch.no_grad():
                masks = model(mix_mag_tensor)
                masks = masks.squeeze(0).cpu().numpy() # (4, 2, Freq, Time)

            # 타겟 마스크 가져오기
            stem_indices = {'vocals': 0, 'drums': 1, 'bass': 2, 'other': 3}
            target_idx = stem_indices.get(args.target, 0)
            mask = masks[target_idx]

            # 복원
            est_mag = mix_mag.cpu().numpy() * mask
            est_audio = processor.stft_to_audio(est_mag, mix_phase)

            # SDR 계산 준비
            # (Channels, Samples) -> (n_src, Samples, Channels)
            ref = ref_audio.T[None, :, :]
            est = est_audio.T[None, :, :]
            
            # 길이 미세 조정
            L = min(ref.shape[1], est.shape[1])
            ref = ref[:, :L, :]
            est = est[:, :L, :]

            # Museval 평가
            # win=L 로 설정하여 곡 전체를 하나의 윈도우로 계산 (Global SDR)
            sdr, _, _, _ = museval.evaluate(ref, est, win=L, hop=L)
            
            # NaN 값 제거 후 중간값 사용
            score = np.nanmedian(sdr)
            sdr_results.append({'song': song_name, 'sdr': score})
            
        except Exception as e:
            print(f"Error evaluating {song_name}: {e}")
            continue

    # 6. 최종 결과 출력
    if not sdr_results:
        print("평가된 곡이 없습니다.")
        return

    # DataFrame으로 보기 좋게 정리
    df = pd.DataFrame(sdr_results)
    mean_sdr = df['sdr'].mean()
    median_sdr = df['sdr'].median()
    
    print("\n" + "="*40)
    print(f"📊 평가 완료: {args.target}")
    print(f"   - 전체 곡 수: {len(df)}")
    print(f"   - 평균 SDR: {mean_sdr:.4f} dB")
    print(f"   - 중앙값 SDR: {median_sdr:.4f} dB")
    print("="*40)
    
    # CSV 저장 (선택)
    df.to_csv(f"eval_results_{args.target}.csv", index=False)
    print(f"상세 결과 저장됨: eval_results_{args.target}.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 폴더 경로 입력 (test 폴더 통째로 넣어도 됨)
    parser.add_argument('--test_dir', type=str, required=True, help='Path to test dataset folder')
    parser.add_argument('--target', type=str, default='vocals', help='Target name (vocals, drums, bass, other)')
    parser.add_argument('--model_path', type=str, required=True, help='Model checkpoint path')
    
    args = parser.parse_args()
    evaluate_dataset(args)