import torch
from torch.utils.data import DataLoader
import os
import datetime

# 모듈 임포트
from options import get_args
from trainer import Trainer
from models.crnn_separator import CRNN_Separator
from utils.audio_processor import AudioProcessor
from utils.dataset import RemixingDataset

def main():
    # 1. 시작 시간 기록
    start_time = datetime.datetime.now()
    print("="*40)
    print(f"⏰ 학습 시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*40)

    # 2. 설정 불러오기
    args = get_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 3. 데이터셋 준비
    processor = AudioProcessor(sr=args.sr, n_fft=args.n_fft, hop_length=args.hop_length)
    
    print(f"🚀 [SSD] 학습 데이터 로딩: {args.train_dir}")
    train_dataset = RemixingDataset(
        args.train_dir, processor, duration=3.0, remix_prob=0.5
    )
    
    print(f"🐢 [HDD] 검증 데이터 로딩: {args.val_dir}")
    val_dataset = RemixingDataset(
        # 검증 때는 remix_prob=0.0 (섞지 않고 원본 그대로 평가)을 추천하지만
        # 데이터가 부족하면 0.5로 두셔도 됩니다. 여기선 원본 평가를 위해 0.0으로 설정함.
        args.val_dir, processor, duration=3.0, remix_prob=0.0 
    )
    
    print(f"   -> 학습 데이터 수: {len(train_dataset)}개")
    print(f"   -> 검증 데이터 수: {len(val_dataset)}개")
    
    # [핵심] DataLoader 설정 (SSD 성능 극대화)
    # num_workers: CPU 코어 수에 맞춰 설정 (보통 4~8). SSD일 때 효과가 큽니다.
    # pin_memory=True: GPU 전송 속도 향상
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        persistent_workers=True # 윈도우에서 에폭마다 프로세스 재생성 방지 (속도 향상)
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True,
        persistent_workers=True
    )
    
    # 4. 모델 준비 (8채널 출력)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CRNN_Separator(
        input_channels=2, 
        n_bins=args.n_fft // 2 + 1, 
        num_stems=4
    ).to(device)
    
    # 파인튜닝 체크 (기존 코드 유지)
    if args.pretrained_path and os.path.exists(args.pretrained_path):
        print(f"♻️ Fine-tuning: 가중치 로드 시도 ({args.pretrained_path})")
        try:
            checkpoint = torch.load(args.pretrained_path)
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            model.load_state_dict(state_dict, strict=False)
            print("   -> 가중치 로드 성공")
        except Exception as e:
            print(f"   -> 로드 실패 (새로 시작): {e}")

    # 5. 트레이너 실행
    trainer = Trainer(model, train_loader, val_loader, args)
    trainer.fit()

    # 6. 종료 시간
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    print("="*40)
    print(f"⏳ 총 소요 시간: {duration}")
    print("="*40)

if __name__ == '__main__':
    main()