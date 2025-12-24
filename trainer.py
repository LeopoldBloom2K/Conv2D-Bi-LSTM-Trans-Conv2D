import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import numpy as np

# utils 폴더에 EarlyStopping이 있다고 가정합니다.
# 만약 없다면 이 줄을 지우고 fit 함수 내부의 관련 코드를 주석 처리하세요.
from utils.early_stopping import EarlyStopping

class Trainer:
    def __init__(self, model, train_loader, val_loader, args):
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # 1. 디바이스 설정 (GPU 우선)
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        self.model = self.model.to(self.device)
        
        # 2. 최적화 도구 (Adam)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=1e-5)
        
        # 3. 손실 함수 (L1 Loss가 음원 분리에 좋음)
        self.criterion = nn.L1Loss()
        
        # 4. 혼합 정밀도 학습 (메모리 절약 & 속도 향상)
        self.scaler = GradScaler('cuda', enabled=self.use_cuda)
        
        # 5. 로깅 및 저장
        self.writer = SummaryWriter(log_dir=f"runs/{args.exp_name}")
        self.best_model_path = os.path.join(args.checkpoint_dir, f'{args.exp_name}_best.pth')
        
        # Early Stopping (25번 동안 성능 향상 없으면 중단)
        patience_val = getattr(args, 'patience', 25) # args에 없으면 기본값 25
        self.early_stopping = EarlyStopping(patience=patience_val, verbose=True, path=self.best_model_path)

    def train_epoch(self, epoch):
        self.model.train()
        train_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs} [Train]")
        
        for mix, targets in pbar:
            # mix: (Batch, 2, Freq, Time)
            # targets: (Batch, 4, 2, Freq, Time)
            mix, targets = mix.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            
            # Mixed Precision
            with autocast(device_type='cuda', enabled=self.use_cuda):
                # 1. 모델이 마스크 예측 (Batch, 4, 2, Freq, Time)
                masks = self.model(mix)
                
                # 2. 마스크를 믹스에 적용
                # mix는 (Batch, 2, ...) 이므로 (Batch, 1, 2, ...)로 차원을 늘려야
                # (Batch, 4, 2, ...)인 마스크와 곱해짐 (Broadcasting)
                mix_expanded = mix.unsqueeze(1) 
                estimated_sources = mix_expanded * masks
                
                # 3. 정답(targets)과 비교하여 손실 계산
                loss = self.criterion(estimated_sources, targets)
            
            # 역전파 (Backpropagation)
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
                    
                    # 검증 손실 계산
                    loss = self.criterion(estimated_sources, targets)
                    
                val_loss += loss.item()
        return val_loss / len(self.val_loader)

    def fit(self):
        print(f"🚀 학습 시작! (Device: {self.device})")
        print(f"🎯 목표: 4개 악기 동시 분리 (Vocals, Drums, Bass, Other)")
        
        for epoch in range(self.args.epochs):
            # 1. 훈련
            train_loss = self.train_epoch(epoch)
            
            # 2. 검증
            val_loss = self.validate()
            
            print(f"Epoch {epoch+1}: Train Loss {train_loss:.5f} | Val Loss {val_loss:.5f}")
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)

            # 3. 조기 종료 및 모델 저장 체크
            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                print("🛑 Early Stopping 발동! 학습을 종료합니다.")
                break
        
        self.writer.close()
        print(f"✨ 학습 완료. 최적 모델 저장됨: {self.best_model_path}")