import torch
from torch.utils.data import Dataset
import os
import random
import numpy as np

class RemixingDataset(Dataset):
    def __init__(self, data_dir, processor, target_name='vocals', duration=3.0, remix_prob=0.5):
        self.data_dir = data_dir
        self.processor = processor
        self.target_name = target_name
        self.duration = duration
        self.remix_prob = remix_prob
        
        self.sources = {'vocals': [], 'drums': [], 'bass': [], 'other': []}
        self.song_folders = [] 
        
        self._scan_files()
        
    def _scan_files(self):
        print(f"Scanning files in {self.data_dir}...")
        for root, dirs, files in os.walk(self.data_dir):
            has_all = all(f"{stem}.wav" in files for stem in self.sources.keys())
            if has_all:
                self.song_folders.append(root)
        print(f"Dataset Loaded: {len(self.song_folders)} songs found.")

    def __len__(self):
        return len(self.song_folders)

    def load_random_chunk(self, path):
        """
        (2, Time) 형태의 스테레오 오디오를 로드하고 랜덤 크롭/패딩
        """
        audio = self.processor.load_audio(path) # Returns (2, Samples)
        
        time_dim = audio.shape[-1]
        req_samples = int(self.processor.sr * self.duration)
        
        if time_dim < req_samples:
            # 짧으면 패딩 (뒷부분 0 채움) -> ((ch_front, ch_back), (time_front, time_back))
            pad_len = req_samples - time_dim
            audio = np.pad(audio, ((0, 0), (0, pad_len)))
        else:
            # 길면 랜덤 크롭
            start = random.randint(0, time_dim - req_samples)
            audio = audio[:, start : start + req_samples]
            
        return audio

    def __getitem__(self, idx):
        # 1. 원본 데이터 로드 (기본 리믹싱 로직 포함)
        mix, targets = self.load_and_process(idx)
        
        # 2. 데이터 증강 (Augmentation) 적용 - 학습 모드일 때만
        if self.remix_prob > 0: # remix_prob가 0보다 크면 학습 모드로 간주
            # [기법 1] 랜덤 게인 조절 (Volume Scaling)
            # 보컬과 각 악기들의 볼륨을 0.8배 ~ 1.2배 사이로 무작위 조절
            for i in range(targets.shape[0]):
                gain = random.uniform(0.8, 1.2)
                targets[i] = targets[i] * gain
            
            # [기법 2] 채널 뒤집기 (Channel Swap)
            # 스테레오 왼쪽/오른쪽을 50% 확률로 뒤집음
            if random.random() > 0.5:
                mix = torch.flip(mix, [0])
                targets = torch.flip(targets, [1]) # (Stem, Channel, Freq, Time)

            # 증강된 타겟들로 새로운 Mix 생성
            mix = torch.sum(targets, dim=0)

        return mix, targets