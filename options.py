import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Music Source Separation Training (Multi-Target)")
    
    # ==========================================
    # 1. 데이터 경로 설정 (SSD / HDD 분리 전략)
    # ==========================================
    parser.add_argument('--train_dir', type=str, required=True, 
                        help='[SSD 권장] 학습 데이터 폴더 경로 (예: C:/Data/musdb18hq/train)')
    parser.add_argument('--val_dir', type=str, required=True, 
                        help='[HDD 가능] 검증 데이터 폴더 경로 (예: D:/Data/musdb18hq/test)')
    
    # 저장 및 로깅 경로
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', 
                        help='모델 가중치(.pth) 저장 경로')
    parser.add_argument('--exp_name', type=str, default='experiment', 
                        help='텐서보드 실험 이름 및 파일명 접두사')
    parser.add_argument('--pretrained_path', type=str, default=None, 
                        help='파인튜닝 시 불러올 기존 모델 경로 (.pth)')
    
    # ==========================================
    # 2. 학습 하이퍼파라미터 (사용자 요청 반영)
    # ==========================================
    parser.add_argument('--epochs', type=int, default=1000, 
                        help='총 학습 반복 횟수 (기본: 1000)')
    parser.add_argument('--patience', type=int, default=25, 
                        help='Early Stopping 인내 횟수 (기본: 25)')
    
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='배치 사이즈 (SSD 사용 시 16 이상 권장, 메모리 부족 시 줄일 것)')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='학습률 (Learning Rate)')
    
    # ==========================================
    # 3. 오디오 신호 처리 설정 (모델과 일치해야 함)
    # ==========================================
    parser.add_argument('--sr', type=int, default=22050, 
                        help='샘플링 레이트 (Hz)')
    parser.add_argument('--n_fft', type=int, default=1024, 
                        help='STFT 윈도우 크기')
    parser.add_argument('--hop_length', type=int, default=256, 
                        help='STFT 이동 간격')
    
    args = parser.parse_args()
    return args