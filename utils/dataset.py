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
            # 파일이 없을 경우를 대비한 예외 처리 (Slakh 등 활용 시 유용) [권장]
            if os.path.exists(path):
                audio = self.processor.load_audio(path, duration=self.duration)
                mag, _ = self.processor.audio_to_stft(audio)
            else:
                # 파일이 없으면 빈 텐서 생성 (크기는 다른 스템 로드 후 맞춰야 하지만 편의상 생략하거나 0으로 채움)
                # 여기서는 원활한 학습을 위해 모든 파일이 있다고 가정합니다.
                # 만약 파일이 없는 경우가 있다면 별도 처리가 필요합니다.
                pass 
                
            stems.append(mag)

        # targets: (Stem, Channel, Freq, Time)
        targets = torch.stack(stems)
        # mix: 모든 스템을 더함 (2, Freq, Time)
        mix = torch.sum(targets, dim=0)
        
        return mix, targets

    def __getitem__(self, idx):
        # 1. 원본 데이터 로드
        mix, targets = self.load_and_process(idx)
        
        # 2. 데이터 증강 (Augmentation) - 학습 모드(remix_prob > 0)일 때만 실행
        if self.remix_prob > 0:
            
            # [추가됨] Silent Vocal Augmentation (20% 확률)
            # 보컬 트랙을 0으로 만들어 "반주만 있는 구간"을 강제로 학습시킴
            # 효과: 반주가 보컬로 새어나가는(Bleed) 현상을 획기적으로 줄임
            if random.random() < 0.2:
                targets[0] = torch.zeros_like(targets[0])

            # [기법 1] 랜덤 볼륨 스케일링 (Gain Augmentation)
            # 보컬과 각 악기들의 볼륨을 0.7배 ~ 1.3배 사이로 무작위 조절
            for i in range(targets.shape[0]):
                gain = random.uniform(0.7, 1.3)
                targets[i] = targets[i] * gain
            
            # [기법 2] 채널 뒤집기 (Channel Swap)
            # 스테레오 왼쪽/오른쪽을 50% 확률로 뒤집음
            if random.random() > 0.5:
                # mix는 아래에서 다시 계산하므로 targets만 뒤집어도 됨 (일관성 위해 둘 다 처리)
                mix = torch.flip(mix, dims=[0])
                targets = torch.flip(targets, dims=[1]) # (Stem, Channel, Freq, Time)

            # [중요] 증강된 타겟들로 새로운 Mix 생성 (물리적 합산)
            # 위에서 보컬을 0으로 만들었거나 볼륨을 바꿨다면, Mix에도 반영되어야 함
            mix = torch.sum(targets, dim=0)

            # [추가됨] 클리핑 방지 (Normalization)
            # 여러 악기를 합치거나 게인을 높였을 때 1.0을 넘어가면 소리가 깨질 수 있음
            max_val = mix.abs().max()
            if max_val > 1.0:
                scale = 0.99 / max_val
                mix = mix * scale
                targets = targets * scale

        return mix, targets

    def __len__(self):
        return len(self.song_dirs)