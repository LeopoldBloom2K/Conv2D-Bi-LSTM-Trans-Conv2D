import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Music Source Separation Training (Multi-Target)")
    
    # 1. 데이터 경로
    parser.add_argument('--train_dir', type=str, required=True, help='학습 데이터 폴더')
    parser.add_argument('--val_dir', type=str, required=True, help='검증 데이터 폴더')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='모델 저장 경로')
    parser.add_argument('--exp_name', type=str, default='experiment', help='실험 이름')
    parser.add_argument('--pretrained_path', type=str, default=None, help='파인튜닝 경로')
    
    # 2. 학습 하이퍼파라미터 (지옥의 트레이닝 기본값)
    parser.add_argument('--epochs', type=int, default=3000, help='총 학습 에폭')
    parser.add_argument('--patience', type=int, default=100, help='Early Stopping')
    parser.add_argument('--batch_size', type=int, default=16, help='배치 사이즈')
    parser.add_argument('--lr', type=float, default=0.001, help='학습률')
    
    # 3. 오디오 설정
    parser.add_argument('--sr', type=int, default=22050)
    parser.add_argument('--n_fft', type=int, default=1024)
    parser.add_argument('--hop_length', type=int, default=256)
    
    # [추가됨] 4. 모델 구조 설정 (Large 모델용)
    parser.add_argument('--hidden_size', type=int, default=256, 
                        help='LSTM 히든 사이즈 (Medium: 256, Large: 512)')
    parser.add_argument('--num_layers', type=int, default=3, 
                        help='LSTM 레이어 수 (Medium: 3, Large: 4)')
    
    args = parser.parse_args()
    return args