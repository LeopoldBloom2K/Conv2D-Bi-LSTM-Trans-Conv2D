import os
import subprocess
import glob
import re
import time
import sys

# ==============================================================================
# 🎛️ 설정 (원하는 대로 수정하세요)
# ==============================================================================
PYTHON_EXEC = sys.executable  # 현재 파이썬 실행 경로
TRAIN_SCRIPT = "train.py"
EVAL_SCRIPT = "evaluate.py"

# 데이터 경로
TRAIN_DIR = r"data\train"
VAL_DIR = r"data\val"

# 학습 설정
EXP_NAME = "Auto_Loop_Experiment"  # 실험 이름
INITIAL_PRETRAINED = r"checkpoints\crnn_large_finetune_ver1_20251227_171118_best.pth" # 최초 시작 모델
HIDDEN_SIZE = 512
NUM_LAYERS = 4
BATCH_SIZE = 8
LR = 0.0001

# 🔄 자동화 루프 설정
EPOCHS_PER_CYCLE = 50      # 한 번 돌릴 때 진행할 Epoch 수
MAX_CYCLES = 500           # 최대 반복 횟수
TARGET_SDR = 7.0           # 목표 점수

# ==============================================================================

def get_latest_best_model(exp_name):
    """체크포인트 폴더에서 가장 최근에 생성된 best.pth 파일을 찾습니다."""
    search_pattern = os.path.join("checkpoints", f"*{exp_name}*best.pth")
    files = glob.glob(search_pattern)
    
    if not files:
        return None
    
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

def parse_sdr_from_output(output_str):
    """evaluate.py의 출력 로그에서 평균 SDR 점수를 추출합니다."""
    match = re.search(r"평균 SDR:\s*([\-\d\.]+)\s*dB", output_str)
    if match:
        return float(match.group(1))
    return None

def main():
    current_pretrained = INITIAL_PRETRAINED
    print(f"🚀 [Auto Train] 시작합니다. (총 {MAX_CYCLES} 사이클 예정)")
    print(f"📂 초기 모델: {current_pretrained}\n")

    for cycle in range(1, MAX_CYCLES + 1):
        print(f"============================================================")
        print(f"🔄 Cycle {cycle}/{MAX_CYCLES} : 학습 시작 (Epochs: {EPOCHS_PER_CYCLE})")
        print(f"============================================================")
        
        # 1. Train 실행
        train_cmd = [
            PYTHON_EXEC, TRAIN_SCRIPT,
            "--train_dir", TRAIN_DIR,
            "--val_dir", VAL_DIR,
            "--exp_name", f"{EXP_NAME}_Cycle{cycle}", 
            "--hidden_size", str(HIDDEN_SIZE),
            "--num_layers", str(NUM_LAYERS),
            "--batch_size", str(BATCH_SIZE),
            "--lr", str(LR),
            "--epochs", str(EPOCHS_PER_CYCLE),
            "--pretrained_path", current_pretrained
        ]
        
        print(f"RUNNING: {' '.join(train_cmd)}")
        try:
            subprocess.run(train_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ 학습 중 치명적 오류 발생! (Cycle {cycle})")
            break
        
        # 2. 방금 학습한 모델 중 Best 찾기
        latest_model = get_latest_best_model(f"{EXP_NAME}_Cycle{cycle}")
        
        if not latest_model:
            print(f"❌ 오류: Cycle {cycle}의 체크포인트를 찾을 수 없습니다. 중단합니다.")
            break
            
        print(f"\n✅ Cycle {cycle} 학습 완료. 생성된 모델: {latest_model}")
        
        # 3. Evaluate 실행
        print(f"\n📊 Cycle {cycle} 평가 시작...")
        eval_cmd = [
            PYTHON_EXEC, EVAL_SCRIPT,
            "--test_dir", VAL_DIR,
            "--model_path", latest_model,
            "--hidden_size", str(HIDDEN_SIZE),
            "--num_layers", str(NUM_LAYERS)
        ]
        
        # [수정됨] encoding='utf-8' 제거 (Windows 인코딩 호환성), stderr 캡처
        # errors='replace'를 추가하여 인코딩 에러로 인한 멈춤 방지
        result = subprocess.run(eval_cmd, capture_output=True, text=True, errors='replace')
        
        print(result.stdout) # 정상 로그 출력
        
        # [추가됨] 에러가 있다면 반드시 출력
        if result.returncode != 0 or result.stderr:
            print("🚨 [WARNING] 평가 스크립트 에러 출력 (STDERR):")
            print(result.stderr)
        
        # 4. SDR 점수 파싱 및 판단
        sdr_score = parse_sdr_from_output(result.stdout)
        
        if sdr_score is not None:
            print(f"⭐ [Cycle {cycle} 결과] 평균 SDR: {sdr_score} dB")
            
            if sdr_score >= TARGET_SDR:
                print(f"🎉 축하합니다! 목표 점수({TARGET_SDR} dB)를 달성했습니다!")
                print("🚀 더 높은 점수를 위해 다음 사이클로 '증폭' 학습을 계속합니다.")
            else:
                print(f"📉 목표 미달. 현재 모델을 베이스로 다음 사이클 재학습을 진행합니다.")
        else:
            print("⚠️ 경고: SDR 점수를 파싱하지 못했습니다. (위의 에러 로그를 확인하세요)")
            # 에러가 나서 점수가 없더라도, 학습된 모델이 있다면 일단 다음 사이클로 넘어가도록 유지
            # (만약 평가만 실패하고 학습은 잘 된 경우를 대비)
        
        # 5. 다음 사이클 준비 (베이스 모델 교체)
        current_pretrained = latest_model
        print(f"⏭️ 다음 학습은 이 모델에서 이어집니다: {current_pretrained}\n")
        
        time.sleep(5)

    print("\n🏁 모든 자동화 사이클이 완료되었습니다.")

if __name__ == "__main__":
    main()