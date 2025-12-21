import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler 
import os
import argparse
from tqdm import tqdm

from models.crnn_separator import CRNN_Separator
from utils.audio_processor import AudioProcessor
from utils.dataset import RemixingDataset
from utils.early_stopping import EarlyStopping

def train(args):
    # CUDA 사용 가능 여부 체크
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f"Device: {device}")
    
    if not use_cuda:
        print("⚠️ 주의: GPU가 감지되지 않았습니다.")
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(args.checkpoint_dir, 'best_model.pth')

    # 프로세서 및 데이터셋 (SR 22050, 1024 FFT 적용)
    processor = AudioProcessor(sr=22050, n_fft=1024, hop_length=256)
    
    # 데이터셋 로드
    full_dataset = RemixingDataset(
        args.data_dir, 
        processor, 
        target_name=args.target,
        duration=3.0,
        remix_prob=0.5
    )
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    model = CRNN_Separator(n_bins=512).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.L1Loss()
    
    # Scaler 초기화 시 'cuda' 명시 및 GPU 없을 땐 끄기
    scaler = GradScaler('cuda', enabled=use_cuda)

    early_stopping = EarlyStopping(patience=15, verbose=True, path=best_model_path)

    print(f"Start training... (Total: {len(full_dataset)} songs)")

    for epoch in range(args.epochs):
        # --- [Phase 1] Train ---
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        
        for mix, target in pbar:
            mix, target = mix.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            # [수정 3] autocast 최신 문법 적용
            with autocast(device_type='cuda', enabled=use_cuda):
                mask = model(mix)
                pred = mix * mask
                loss = criterion(pred, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_train_loss = train_loss / len(train_loader)

        # --- [Phase 2] Validation ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for mix, target in val_loader:
                mix, target = mix.to(device), target.to(device)
                
                # Validation에서도 동일하게 적용
                with autocast(device_type='cuda', enabled=use_cuda):
                    mask = model(mix)
                    pred = mix * mask
                    loss = criterion(pred, target)
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1} Result: Train Loss {avg_train_loss:.5f} | Val Loss {avg_val_loss:.5f}")

        # --- [Phase 3] Early Stopping Check (수정됨) ---
        early_stopping(avg_val_loss, model)
        
        if early_stopping.early_stop:
            # [핵심] 현재 Epoch가 50 미만이면 강제로 멈추지 않게 함
            if epoch + 1 < 50:
                print(f"⏳ 최소 50 Epoch 보장을 위해 Early Stopping을 미룹니다. (현재: {epoch+1})")
                # 카운터와 플래그를 리셋해서 계속 학습하게 만듦
                early_stopping.early_stop = False
                early_stopping.counter = 0
            else:
                print(f"🛑 조기 종료됨! (Epoch {epoch+1})")
                print("성능이 더 이상 개선되지 않아 학습을 멈춥니다.")
                break
            
    print(f"학습 종료. 최고 성능 모델 저장됨: {best_model_path}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 경로 기본값 설정됨
    parser.add_argument('--data_dir', type=str, default='D:\\musdb18hq\\train')
    parser.add_argument('--target', type=str, default='vocals')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    
    args = parser.parse_args()
    train(args)