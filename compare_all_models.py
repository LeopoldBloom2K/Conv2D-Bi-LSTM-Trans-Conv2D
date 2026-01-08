import os
import glob
import subprocess
import re
import pandas as pd
import sys

# =========================================================
# 🎛️ 설정
# =========================================================
PYTHON_EXEC = sys.executable
EVAL_SCRIPT = "evaluate.py"
VAL_DIR = r"data\val"     # 평가할 데이터셋 경로
CHECKPOINT_DIR = "check_model" # 모델들이 저장된 폴더
HIDDEN_SIZE = 512
NUM_LAYERS = 4
# =========================================================

def parse_sdr(output):
    """로그에서 SDR 점수 추출"""
    match = re.search(r"평균 SDR:\s*([\-\d\.]+)\s*dB", output)
    if match:
        return float(match.group(1))
    return None

def main():
    # 1. 모든 best.pth 파일 찾기
    models = glob.glob(os.path.join(CHECKPOINT_DIR, "*best.pth"))
    models.sort(key=os.path.getmtime, reverse=True) # 최신순 정렬

    if not models:
        print("❌ 평가할 모델 파일(.pth)이 없습니다.")
        return

    results = []
    print(f"🔍 총 {len(models)}개의 모델을 비교합니다...\n")

    for i, model_path in enumerate(models):
        model_name = os.path.basename(model_path)
        print(f"[{i+1}/{len(models)}] 평가 중: {model_name}")

        cmd = [
            PYTHON_EXEC, EVAL_SCRIPT,
            "--test_dir", VAL_DIR,
            "--model_path", model_path,
            "--hidden_size", str(HIDDEN_SIZE),
            "--num_layers", str(NUM_LAYERS)
        ]

        # 평가 실행 (에러 무시하고 진행)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, errors='replace')
            sdr = parse_sdr(proc.stdout)
            
            if sdr is not None:
                print(f"   👉 SDR: {sdr} dB")
                results.append({"Model": model_name, "SDR": sdr})
            else:
                print("   ⚠️ 점수 파싱 실패")
                if proc.stderr:
                    print(f"   [Error Log] {proc.stderr[:200]}...") # 에러 일부 출력
        except Exception as e:
            print(f"   ❌ 실행 오류: {e}")

    # 2. 결과 출력 (랭킹)
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values(by="SDR", ascending=False).reset_index(drop=True)
        
        print("\n" + "="*50)
        print("🏆 모델 SDR 리더보드 🏆")
        print("="*50)
        print(df)
        print("="*50)
        
        # CSV로 저장
        df.to_csv("model_leaderboard.csv", index=False)
        print("💾 'model_leaderboard.csv' 파일로 저장되었습니다.")
    else:
        print("\n❌ 평가 결과가 없습니다.")

if __name__ == "__main__":
    main()