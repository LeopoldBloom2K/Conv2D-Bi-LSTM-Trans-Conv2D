import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import numpy as np

# utils 폴더에 EarlyStopping이 있다고 가정합니다.
from utils.early_stopping import EarlyStopping

class Trainer:
    def __init__(self, model, train_loader, val_loader, args):
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # 1. 디바이스 설정
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        self.model = self.model.to(self.device)
        
        # 2. 최적화 도구
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=1e-5)
        
        # 3. 손실 함수
        self.criterion = nn.L1Loss()
        
        # 4. 혼합 정밀도 학습
        self.scaler = GradScaler('cuda', enabled=self.use_cuda)
        
        # 5. 로깅 및 저장 [수정됨: 에러 방지용 안전 코드]
        # 윈도우 경로 호환성을 위해 os.path.join 사용
        log_dir = os.path.join("runs", args.exp_name)
        # 폴더가 없으면 미리 생성 (Tensorboard 에러 방지)
        os.makedirs(log_dir, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=log_dir)
        self.best_model_path = os.path.join(args.checkpoint_dir, f'{args.exp_name}_best.pth')
        
        # Early Stopping
        patience_val = getattr(args, 'patience', 25)
        self.best_score = None 
        # (Trainer 내부에서만 쓸 간단한 변수, 혹은 utils.EarlyStopping 사용)
        self.early_stopping = EarlyStopping(patience=patience_val, verbose=True, path=self.best_model_path)

    def train_epoch(self, epoch):
        self.model.train()
        train_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs} [Train]")
        
        for mix, targets in pbar:
            mix, targets = mix.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            
            with autocast(device_type='cuda', enabled=self.use_cuda):
                masks = self.model(mix)
                mix_expanded = mix.unsqueeze(1) 
                estimated_sources = mix_expanded * masks
                loss = self.criterion(estimated_sources, targets)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        return train_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for mix, targets in self.val_loader:
                mix, targets = mix.to(self.device), targets.to(self.device)
                
                with autocast(device_type='cuda', enabled=self.use_cuda):
                    masks = self.model(mix)
                    mix_expanded = mix.unsqueeze(1)
                    estimated_sources = mix_expanded * masks
                    loss = self.criterion(estimated_sources, targets)
                    
                val_loss += loss.item()
        return val_loss / len(self.val_loader)

    def fit(self):
        print(f"🚀 학습 시작! (Device: {self.device})")
        print(f"🎯 목표: 4개 악기 동시 분리")
        
        for epoch in range(self.args.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            
            print(f"Epoch {epoch+1}: Train Loss {train_loss:.5f} | Val Loss {val_loss:.5f}")
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)

            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                print("🛑 Early Stopping 발동! 학습을 종료합니다.")
                break
        
        self.writer.close()
        print(f"✨ 학습 완료. 최적 모델 저장됨: {self.best_model_path}")