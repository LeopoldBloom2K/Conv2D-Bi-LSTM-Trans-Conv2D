import stempeg
import os
import soundfile as sf
import numpy as np
from tqdm import tqdm
import glob

# ================= 설정 =================
# 1. 다운로드 받은 MUSDB18 원본 폴더 (stem.mp4 파일들이 있는 곳)
SOURCE_DIR = r"data/raw/musdb18/train" 

# 2. 변환된 wav 파일이 저장될 폴더 (우리 학습 코드에서 쓸 경로)
DEST_DIR = r"data/processed/musdb18/train"
# =======================================

def convert_musdb():
    if not os.path.exists(SOURCE_DIR):
        print(f"❌ 원본 경로를 찾을 수 없습니다: {SOURCE_DIR}")
        return

    # .stem.mp4 파일 찾기
    stem_files = glob.glob(os.path.join(SOURCE_DIR, "*.stem.mp4"))
    
    print(f"총 {len(stem_files)}개의 곡을 변환합니다...")
    os.makedirs(DEST_DIR, exist_ok=True)

    for stem_path in tqdm(stem_files):
        try:
            # 줄기(Stem) 읽기 (0:mix, 1:drums, 2:bass, 3:other, 4:vocals)
            # MUSDB18 포맷: (stems, samples, channels)
            # stempeg는 기본적으로 float32로 로드함
            stems, rate = stempeg.read_stems(stem_path)
            
            # 곡 이름 추출
            song_name = os.path.splitext(os.path.basename(stem_path))[0]
            song_folder = os.path.join(DEST_DIR, song_name)
            os.makedirs(song_folder, exist_ok=True)
            
            # 저장할 파일 매핑
            # stempeg 로드 순서: [mixture, drums, bass, other, vocals]
            mapping = {
                0: "mixture.wav",
                1: "drums.wav",
                2: "bass.wav",
                3: "other.wav",
                4: "vocals.wav"
            }
            
            for idx, fname in mapping.items():
                audio = stems[idx] # (samples, 2)
                out_path = os.path.join(song_folder, fname)
                sf.write(out_path, audio, rate)
                
        except Exception as e:
            print(f"\n⚠️ 에러 발생 ({stem_path}): {e}")

    print("\n✅ 변환 완료!")
    print(f"학습 시 사용할 경로: {DEST_DIR}")

if __name__ == '__main__':
    convert_musdb()