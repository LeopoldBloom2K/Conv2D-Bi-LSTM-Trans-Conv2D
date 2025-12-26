import torch
from torch.utils.data import DataLoader
import os
import datetime

from options import get_args
from trainer import Trainer
from models.crnn_separator import CRNN_Separator
from utils.audio_processor import AudioProcessor
from utils.dataset import RemixingDataset

def main():
    start_time = datetime.datetime.now()
    timestamp = start_time.strftime('%Y%m%d_%H%M%S')
    
    print("="*40)
    print(f"⏰ 학습 시작: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*40)

    args = get_args()
    args.exp_name = f"{args.exp_name}_{timestamp}"
    
    print(f"📝 실험 이름: {args.exp_name}")
    print(f"⚙️ 모델 설정: Hidden={args.hidden_size}, Layers={args.num_layers}")
    
    # 체크포인트 폴더 생성 (필수)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    processor = AudioProcessor(sr=args.sr, n_fft=args.n_fft, hop_length=args.hop_length)
    
    train_dataset = RemixingDataset(args.train_dir, processor, duration=3.0, remix_prob=0.5)
    val_dataset = RemixingDataset(args.val_dir, processor, duration=3.0, remix_prob=0.0)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # [수정] 옵션값 전달하여 모델 생성
    model = CRNN_Separator(
        input_channels=2, 
        n_bins=args.n_fft // 2, 
        num_stems=4,
        hidden_size=args.hidden_size, # 옵션 적용
        num_layers=args.num_layers    # 옵션 적용
    ).to(device)
    
    if args.pretrained_path and os.path.exists(args.pretrained_path):
        print(f"♻️ Fine-tuning: {args.pretrained_path}")
        try:
            checkpoint = torch.load(args.pretrained_path)
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            model.load_state_dict(state_dict, strict=False)
            print("   -> 로드 성공")
        except Exception as e:
            print(f"   -> 로드 실패: {e}")

    trainer = Trainer(model, train_loader, val_loader, args)
    trainer.fit()

    end_time = datetime.datetime.now()
    print(f"⏳ 소요 시간: {end_time - start_time}")

if __name__ == '__main__':
    main()