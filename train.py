import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.crnn_separator import CRNN_Separator
from utils.audio_processor import AudioProcessor
from utils.dataset import MusicSeparationDataset
import os
import argparse
from tqdm import tqdm

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 1. Setup
    processor = AudioProcessor(n_fft=2048, hop_length=512)
    dataset = MusicSeparationDataset(
        args.data_dir, 
        processor, 
        target_name=args.target,
        duration=3.0
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    model = CRNN_Separator().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss() # L1 Loss가 오디오 스펙트로그램 복원에 더 선명함

    # 2. Fine-tuning / Resume Logic
    start_epoch = 0
    if args.resume_from:
        if os.path.isfile(args.resume_from):
            print(f"loading checkpoint: {args.resume_from}")
            checkpoint = torch.load(args.resume_from)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            print(f"Resuming from epoch {start_epoch}")
        else:
            print("Checkpoint not found, starting from scratch.")

    # 3. Training Loop
    print(f"Start training for target: {args.target}")
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for mix, target in pbar:
            mix, target = mix.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            mask = model(mix)
            predicted_mag = mix * mask # 마스킹 적용
            
            loss = criterion(predicted_mag, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
        
        # Save Checkpoint
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(args.checkpoint_dir, f"model_ep{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path)
            print(f"Saved: {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to training data')
    parser.add_argument('--target', type=str, default='vocals', help='Target instrument (e.g. vocals, drums)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to .pth file for fine-tuning')
    
    args = parser.parse_args()
    train(args)