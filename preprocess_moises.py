import os
import glob
import soundfile as sf
import numpy as np
from tqdm import tqdm

# ================= 설정 (경로 확인 필수!) =================
# 1. MoisesDB 최상위 폴더
SOURCE_DIR = r"D:\moisesdb\moisesdb_v0.1" 

# 2. 변환된 데이터가 저장될 경로
DEST_DIR = r"data\raw\moisesdb\train"
# ========================================================

def load_and_sum_audio_in_folder(folder_path):
    """
    해당 폴더 안에 있는 '모든' wav/flac 파일을 찾아서 합칩니다.
    파일이 여러 개(guitar1, guitar2...)여도 누락 없이 다 더합니다.
    """
    if not os.path.exists(folder_path):
        return None, None
    
    # 폴더 내 모든 오디오 파일 검색 (하위 폴더 포함하지 않음, 현재 폴더만)
    files = glob.glob(os.path.join(folder_path, "*.wav")) + glob.glob(os.path.join(folder_path, "*.flac"))
    
    if not files:
        return None, None
    
    # 첫 번째 파일로 초기화
    combined_audio, sr = sf.read(files[0])
    
    # 두 번째 파일부터는 더하기 (Merge)
    if len(files) > 1:
        for f in files[1:]:
            audio, _ = sf.read(f)
            # 길이 맞추기 (짧은 쪽에 맞춤)
            min_len = min(len(combined_audio), len(audio))
            combined_audio = combined_audio[:min_len] + audio[:min_len]
            
    return combined_audio, sr

def preprocess():
    if not os.path.exists(SOURCE_DIR):
        print(f"❌ 원본 경로를 찾을 수 없습니다: {SOURCE_DIR}")
        return

    # UUID로 된 노래 폴더들 목록
    song_folders = [f.path for f in os.scandir(SOURCE_DIR) if f.is_dir()]
    print(f"총 {len(song_folders)}개의 곡 폴더를 발견했습니다. 안전 변환 시작...")
    
    os.makedirs(DEST_DIR, exist_ok=True)
    success_count = 0

    for folder in tqdm(song_folders):
        try:
            song_uuid = os.path.basename(folder)
            target_folder = os.path.join(DEST_DIR, song_uuid)
            
            # 데이터 담을 변수들
            final_stems = {
                'vocals': None, 'drums': None, 'bass': None, 'other': []
            }
            sample_rate = None

            # --- 폴더 내 모든 하위 폴더(악기들) 스캔 ---
            sub_dirs = [d.name for d in os.scandir(folder) if d.is_dir()]
            
            for sub in sub_dirs:
                sub_path = os.path.join(folder, sub)
                sub_lower = sub.lower()
                
                audio, sr = load_and_sum_audio_in_folder(sub_path)
                
                if audio is None: continue
                if sample_rate is None: sample_rate = sr
                
                # 채널 수 맞추기 (Mono -> Stereo 강제 변환)
                # 간혹 특정 악기가 모노일 경우 합칠 때 에러가 나므로 (N, 2)로 통일
                if audio.ndim == 1:
                    audio = np.column_stack([audio, audio])
                
                # 1. 필수 악기 분류
                if 'vocals' in sub_lower or 'vocal' in sub_lower:
                    if final_stems['vocals'] is None:
                        final_stems['vocals'] = audio
                    else:
                        min_len = min(len(final_stems['vocals']), len(audio))
                        final_stems['vocals'] = final_stems['vocals'][:min_len] + audio[:min_len]

                elif 'drums' in sub_lower or 'drum' in sub_lower:
                    if final_stems['drums'] is None:
                        final_stems['drums'] = audio
                    else:
                        min_len = min(len(final_stems['drums']), len(audio))
                        final_stems['drums'] = final_stems['drums'][:min_len] + audio[:min_len]

                elif 'bass' in sub_lower:
                    if final_stems['bass'] is None:
                        final_stems['bass'] = audio
                    else:
                        min_len = min(len(final_stems['bass']), len(audio))
                        final_stems['bass'] = final_stems['bass'][:min_len] + audio[:min_len]

                # 2. 나머지 모든 악기 -> Other로 분류
                else:
                    final_stems['other'].append(audio)

            # 필수 트랙 확인
            if final_stems['vocals'] is None or final_stems['drums'] is None or final_stems['bass'] is None:
                continue

            # --- 저장 준비 ---
            os.makedirs(target_folder, exist_ok=True)
            
            # 전체 트랙 중 가장 짧은 길이에 맞춤
            min_len = min(len(final_stems['vocals']), len(final_stems['drums']), len(final_stems['bass']))
            for o in final_stems['other']:
                min_len = min(min_len, len(o))
                
            def crop(arr): return arr[:min_len]

            v_final = crop(final_stems['vocals'])
            d_final = crop(final_stems['drums'])
            b_final = crop(final_stems['bass'])
            
            # Other 합치기
            if final_stems['other']:
                # 첫 번째 other 트랙을 기준으로 형상 초기화
                o_final = np.zeros_like(v_final)
                for o in final_stems['other']:
                    cropped_o = crop(o)
                    # 혹시 모를 채널 수 불일치 방지 (이미 위에서 처리했지만 이중 안전장치)
                    if cropped_o.shape == o_final.shape:
                        o_final += cropped_o
            else:
                o_final = np.zeros_like(v_final)

            # Mixture 생성
            mixture = v_final + d_final + b_final + o_final
            
            # --- [중요 수정] 정규화 (Normalization) ---
            # Mixture가 클리핑(1.0 초과)되면, '비율(Scale Factor)'을 구해서 모든 트랙을 똑같이 줄여야 함
            max_val = np.max(np.abs(mixture))
            if max_val > 1.0:
                scale_factor = 0.99 / max_val # 0.99로 안전하게 줄임
                mixture *= scale_factor
                v_final *= scale_factor
                d_final *= scale_factor
                b_final *= scale_factor
                o_final *= scale_factor

            # 파일 저장
            # float32 형변환 (용량 절약 및 호환성)
            sf.write(os.path.join(target_folder, "vocals.wav"), v_final.astype(np.float32), sample_rate)
            sf.write(os.path.join(target_folder, "drums.wav"), d_final.astype(np.float32), sample_rate)
            sf.write(os.path.join(target_folder, "bass.wav"), b_final.astype(np.float32), sample_rate)
            sf.write(os.path.join(target_folder, "other.wav"), o_final.astype(np.float32), sample_rate)
            sf.write(os.path.join(target_folder, "mixture.wav"), mixture.astype(np.float32), sample_rate)
            
            success_count += 1
            
        except Exception as e:
            print(f"\nError processing {folder}: {e}")
            continue

    print("\n✅ 안전 변환 완료!")
    print(f"총 {success_count}개의 곡이 변환되었습니다.")

if __name__ == '__main__':
    preprocess()