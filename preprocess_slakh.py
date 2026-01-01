import os
import yaml  # [변경] json 대신 yaml 사용
import soundfile as sf
import numpy as np
import librosa
from tqdm import tqdm
import glob

# ================= 사용자 환경 설정 =================
# 1. Slakh2100 데이터셋의 루트 폴더
SOURCE_DIR = r"D:\slakh2100_flac_redux"

# 2. 변환할 서브셋
SUBSETS = ['train'] 

# 3. 변환된 데이터가 저장될 경로
DEST_DIR = r"D:\slakh2100_preprocessed\train"

# 4. 모델 학습용 샘플링 레이트
TARGET_SR = 22050
# ======================================================

def preprocess_slakh():
    if not os.path.exists(SOURCE_DIR):
        print(f"❌ 원본 경로를 찾을 수 없습니다: {SOURCE_DIR}")
        return

    track_folders = []
    for subset in SUBSETS:
        subset_path = os.path.join(SOURCE_DIR, subset)
        tracks = glob.glob(os.path.join(subset_path, "Track*"))
        track_folders.extend(tracks)
    
    print(f"📂 총 {len(track_folders)}개의 곡을 변환합니다... (대상 폴더: {SUBSETS})")
    os.makedirs(DEST_DIR, exist_ok=True)
    
    success_count = 0

    for track_path in tqdm(track_folders):
        try:
            track_id = os.path.basename(track_path)
            
            # [변경] metadata.yaml 파일 로드
            metadata_path = os.path.join(track_path, "metadata.yaml")
            
            if not os.path.exists(metadata_path):
                # 혹시나 json이 섞여 있을 경우를 대비한 예외 처리
                metadata_path_json = os.path.join(track_path, "metadata.json")
                if os.path.exists(metadata_path_json):
                     import json
                     with open(metadata_path_json, 'r') as f:
                        metadata = json.load(f)
                else:
                    continue
            else:
                # YAML 파일 읽기
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = yaml.safe_load(f)

            stems_map = {'drums': [], 'bass': [], 'other': []}

            # 스템 정보 파싱
            # YAML 구조도 JSON과 동일하게 stems 키 밑에 정보가 있다고 가정
            if 'stems' not in metadata:
                continue

            for stem_key, stem_info in metadata['stems'].items():
                inst_class = stem_info.get('inst_class')
                
                # 파일명: Key값 + .flac (예: S00.flac)
                file_name = f"{stem_key}.flac"
                file_path = os.path.join(track_path, "stems", file_name)
                
                if not os.path.exists(file_path):
                    # wav일 경우 대비
                    file_path_wav = os.path.join(track_path, "stems", f"{stem_key}.wav")
                    if os.path.exists(file_path_wav):
                        file_path = file_path_wav
                    else:
                        continue

                # 오디오 로드
                audio, _ = librosa.load(file_path, sr=TARGET_SR, mono=False)
                
                if audio.ndim == 1:
                    audio = np.stack([audio, audio], axis=0)

                # 악기 분류
                if inst_class == 'Drums':
                    stems_map['drums'].append(audio)
                elif inst_class == 'Bass':
                    stems_map['bass'].append(audio)
                else:
                    stems_map['other'].append(audio)

            if not any(stems_map.values()):
                continue
                
            # 길이 맞추기
            max_len = 0
            for group in stems_map.values():
                for audio in group:
                    max_len = max(max_len, audio.shape[1])

            # 합치기
            def sum_stems(stem_list, length):
                if not stem_list:
                    return np.zeros((2, length), dtype=np.float32)
                
                mix_result = np.zeros((2, length), dtype=np.float32)
                for audio in stem_list:
                    curr_len = audio.shape[1]
                    mix_result[:, :curr_len] += audio
                return mix_result

            final_drums = sum_stems(stems_map['drums'], max_len)
            final_bass = sum_stems(stems_map['bass'], max_len)
            final_other = sum_stems(stems_map['other'], max_len)
            
            # [핵심] Vocals는 0 (Silence)
            final_vocals = np.zeros((2, max_len), dtype=np.float32)

            # Mixture 생성
            final_mixture = final_vocals + final_drums + final_bass + final_other

            # 클리핑 방지
            max_val = np.max(np.abs(final_mixture))
            if max_val > 1.0:
                scale = 0.99 / max_val
                final_mixture *= scale
                final_vocals *= scale
                final_drums *= scale
                final_bass *= scale
                final_other *= scale

            # 저장
            out_folder = os.path.join(DEST_DIR, track_id)
            os.makedirs(out_folder, exist_ok=True)
            
            sf.write(os.path.join(out_folder, "vocals.wav"), final_vocals.T, TARGET_SR)
            sf.write(os.path.join(out_folder, "drums.wav"), final_drums.T, TARGET_SR)
            sf.write(os.path.join(out_folder, "bass.wav"), final_bass.T, TARGET_SR)
            sf.write(os.path.join(out_folder, "other.wav"), final_other.T, TARGET_SR)
            sf.write(os.path.join(out_folder, "mixture.wav"), final_mixture.T, TARGET_SR)

            success_count += 1
            
        except Exception as e:
            print(f"⚠️ Error processing {track_path}: {e}")
            continue

    print("\n✅ Slakh2100 변환 완료!")
    print(f"   - 저장 경로: {DEST_DIR}")
    print(f"   - 변환된 곡 수: {success_count} / {len(track_folders)}")

if __name__ == '__main__':
    preprocess_slakh()