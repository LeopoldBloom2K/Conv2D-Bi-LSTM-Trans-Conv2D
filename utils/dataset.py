import torch
from torch.utils.data import Dataset
import os
import random
import numpy as np

class RemixingDataset(Dataset):
    def __init__(self, data_dir, processor, duration=3.0, remix_prob=0.5):
        self.data_dir = data_dir
        self.processor = processor
        self.duration = duration
        self.remix_prob = remix_prob
        # 각 곡의 폴더 리스트 확보
        self.song_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) 
                          if os.path.isdir(os.path.join(data_dir, d))]

    def load_and_process(self, idx):
        """기존의 핵심 로직: 파일을 읽고 스펙트로그램으로 변환"""
        song_dir = self.song_dirs[idx]
        stem_names = ['vocals', 'drums', 'bass', 'other']
        stems = []

        # 각 스템 로드
        for name in stem_names:
            path = os.path.join(song_dir, f"{name}.wav")
            audio = self.processor.load_audio(path, duration=self.duration)
            mag, _ = self.processor.audio_to_stft(audio)
            stems.append(mag)

        # targets: (Stem, Channel, Freq, Time)
        targets = torch.stack(stems)
        # mix: 모든 스템을 더함 (2, Freq, Time)
        mix = torch.sum(targets, dim=0)
        
        return mix, targets

    def __getitem__(self, idx):
        # 1. 원본 데이터 로드
        mix, targets = self.load_and_process(idx)
        
        # # 2. 데이터 증강 (Augmentation) - 학습 모드(remix_prob > 0)일 때만 실행
        # if self.remix_prob > 0:
        #     # [기법 1] 랜덤 볼륨 스케일링 (Gain Augmentation)
        #     # 보컬과 각 악기들의 볼륨을 0.7배 ~ 1.3배 사이로 무작위 조절 (더 넓은 범위 권장)
        #     for i in range(targets.shape[0]):
        #         gain = random.uniform(0.7, 1.3)
        #         targets[i] = targets[i] * gain
            
        #     # [기법 2] 채널 뒤집기 (Channel Swap)
        #     # 스테레오 왼쪽/오른쪽을 50% 확률로 뒤집음
        #     if random.random() > 0.5:
        #         # mix: (Channel, Freq, Time), targets: (Stem, Channel, Freq, Time)
        #         mix = torch.flip(mix, dims=[0])
        #         targets = torch.flip(targets, dims=[1])

        #     # 증강된 타겟들로 새로운 Mix 생성 (물리적 합산)
        #     mix = torch.sum(targets, dim=0)

        return mix, targets

    def __len__(self):
        return len(self.song_dirs)