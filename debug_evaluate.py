import os
import glob
import subprocess
import sys

# =========================================================
# 🎛️ 설정 (경로를 확인하세요!)
# =========================================================
PYTHON_EXEC = sys.executable
EVAL_SCRIPT = "evaluate.py"
VAL_DIR = r"data\val"           # 검증 데이터 경로
CHECKPOINT_DIR = "check_model"  # ⚠️ 사용자님이 말씀하신 폴더명으로 수정함
HIDDEN_SIZE = 512
NUM_LAYERS = 4
# =========================================================

def main():
    # 1. 파일 찾기
    if not os.path.exists(CHECKPOINT_DIR):
        print(f"❌ 오류: '{CHECKPOINT_DIR}' 폴더가 없습니다. 경로를 확인해주세요.")
        return

    # 모든 .pth 파일 찾기
    models = glob.glob(os.path.join(CHECKPOINT_DIR, "*best.pth"))
    
    if not models:
        print(f"❌ 오류: '{CHECKPOINT_DIR}' 폴더 안에 '*best.pth' 파일이 하나도 없습니다.")
        return

    # 2. Cycle1 파일이 있는지 확인하고 맨 앞으로 가져오기 (사용자 요청 반영)
    cycle1_files = [m for m in models if "Cycle1_" in m]
    other_files = [m for m in models if "Cycle1_" not in m]
    
    # 최신순 정렬 (나머지 파일들)
    other_files.sort(key=os.path.getmtime, reverse=True)
    
    # Cycle1 우선 평가 리스트 생성
    sorted_models = cycle1_files + other_files
    
    print(f"🔍 총 {len(sorted_models)}개의 모델을 찾았습니다.")
    print(f"👉 첫 번째 평가 대상: {os.path.basename(sorted_models[0])}")
    print("="*60)

    # 3. 평가 실행 (에러 숨기지 않음)
    for i, model_path in enumerate(sorted_models):
        model_name = os.path.basename(model_path)
        print(f"\n▶ [{i+1}/{len(sorted_models)}] 평가 시작: {model_name}")
        print(f"   파일 경로: {model_path}")
        
        cmd = [
            PYTHON_EXEC, EVAL_SCRIPT,
            "--test_dir", VAL_DIR,
            "--model_path", model_path,
            "--hidden_size", str(HIDDEN_SIZE),
            "--num_layers", str(NUM_LAYERS)
        ]

        # subprocess 호출 시 capture_output=False로 설정하여 
        # 에러 메시지가 터미널에 직접 출력되게 함
        try:
            exit_code = subprocess.call(cmd)
            
            if exit_code != 0:
                print(f"\n🚨 [CRITICAL ERROR] 평가 스크립트가 에러 코드 {exit_code}로 종료되었습니다.")
                print("위의 Traceback 메시지를 확인해주세요!")
                break # 첫 번째 에러에서 멈춤 (원인 분석을 위해)
            else:
                print("✅ 평가 완료.")
                
        except Exception as e:
            print(f"❌ 실행 중 예외 발생: {e}")
            break

if __name__ == "__main__":
    main()