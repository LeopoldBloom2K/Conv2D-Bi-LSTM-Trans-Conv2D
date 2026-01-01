import torch
import os

def merge_models(path1, path2, save_path):
    print(f"🔄 모델 병합 시작...")
    
    # 모델 로드
    ckpt1 = torch.load(path1, map_location='cpu')
    ckpt2 = torch.load(path2, map_location='cpu')

    # 가중치 딕셔너리 추출
    state_dict1 = ckpt1['model_state_dict'] if 'model_state_dict' in ckpt1 else ckpt1
    state_dict2 = ckpt2['model_state_dict'] if 'model_state_dict' in ckpt2 else ckpt2

    # 새로운 가중치를 담을 딕셔너리
    merged_dict = {}

    # 모든 레이어를 돌며 평균 계산
    for key in state_dict1.keys():
        if key in state_dict2:
            # 두 모델의 가중치를 5:5로 평균 (0.5, 0.5)
            # 만약 성능이 더 좋은 모델에 비중을 더 주고 싶다면 (0.4, 0.6) 식으로 조절 가능
            merged_dict[key] = (state_dict1[key] * 0.8) + (state_dict2[key] * 0.2)
        else:
            print(f"⚠️ 경고: {key} 가 두 번째 모델에 없습니다. 첫 번째 모델 값을 사용합니다.")
            merged_dict[key] = state_dict1[key]

    # 저장
    torch.save({'model_state_dict': merged_dict}, save_path)
    print(f"✅ 병합 완료! 저장된 경로: {save_path}")

if __name__ == "__main__":
    # 파일 경로를 사용자님의 환경에 맞게 수정하세요
    MODEL_A = "checkpoints/crnn_large_ultimate_4db_20251228_195447_best.pth"
    MODEL_B = "checkpoints/crnn_large_final_tune_20251228_063127_best.pth" # 3.93dB
    OUTPUT = "checkpoints/crnn_large_merged_0.9_0.1.pth"

    merge_models(MODEL_A, MODEL_B, OUTPUT)