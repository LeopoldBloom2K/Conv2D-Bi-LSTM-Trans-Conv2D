import os
import torch
from utils.audio_processor import AudioProcessor
from utils.dataset import RemixingDataset

# ==========================================
# [사용자 설정] 실제 학습 데이터 경로 중 하나만 적어주세요
TEST_DIR = r"data\train"  # 예: "data/musdb18/train"
# ==========================================

def debug_dataset():
    print(f"🔍 데이터 경로 점검: {TEST_DIR}")
    
    if not os.path.exists(TEST_DIR):
        print("❌ 경로가 존재하지 않습니다.")
        return

    # 1. 실제 폴더 내부 파일명 확인
    subfolders = [d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))]
    if not subfolders:
        print("❌ 하위 폴더가 없습니다.")
        return
    
    sample_folder = os.path.join(TEST_DIR, subfolders[0])
    print(f"\n📂 첫 번째 샘플 폴더 분석: {sample_folder}")
    print("   [실제 존재하는 파일 목록]")
    files = os.listdir(sample_folder)
    for f in files:
        if f.endswith(".wav"):
            print(f"    - {f}")

    # 2. Dataset 클래스가 어떻게 읽는지 확인
    print("\n🕵️ Dataset 로딩 시뮬레이션")
    processor = AudioProcessor(sr=44100, n_fft=2048, hop_length=1024)
    
    # 우리가 강제한 타겟 순서
    TARGET_STEMS = ['vocals', 'drums', 'bass', 'other']
    print(f"   👉 코드의 타겟 명칭: {TARGET_STEMS}")

    ds = RemixingDataset(
        TEST_DIR, 
        processor, 
        duration=3.0, 
        remix_prob=0.0, 
        target_stems=TARGET_STEMS
    )
    
    # 데이터 하나 로드 시도
    try:
        mix, targets = ds[0]
        print("\n✅ 데이터 로드 성공 (Shape 확인)")
        print(f"   - Mix Shape: {mix.shape}")
        print(f"   - Targets Shape: {targets.shape} (Stem, Channel, Freq, Time)")
        
        # 각 스템별 최대 볼륨 확인 (0이면 로드 안 된 것)
        print("\n📊 스템별 신호 강도 (Max Value):")
        for i, name in enumerate(TARGET_STEMS):
            max_val = targets[i].max().item()
            status = "🔴 0 (로드 실패/무음)" if max_val == 0 else f"🟢 {max_val:.4f}"
            print(f"   [{i}] {name}: {status}")
            
            # 경고 메시지
            if max_val == 0:
                print(f"      ⚠️ 경고: '{name}' 파일이 없거나 이름이 달라서 0으로 채워졌습니다!")
                
    except Exception as e:
        print(f"\n❌ 로드 중 에러 발생: {e}")

if __name__ == '__main__':
    debug_dataset()