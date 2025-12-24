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
        track_path = self.song_folders[idx]
        
        # 1. 4개 스템 로드 (Vocals, Drums, Bass, Other)
        stem_names = ['vocals', 'drums', 'bass', 'other']
        stems = []
        
        for name in stem_names:
            wav_path = os.path.join(track_path, f"{name}.wav")
            y = self.load_random_chunk(wav_path) # (2, Time)
            stems.append(y)
            
        # 2. 스택 쌓기 -> (4, 2, Time)
        sources = np.stack(stems)
        
        # 3. 믹스 생성 (모든 소스 합산) -> (2, Time)
        mix = sources.sum(axis=0) 
        
        # 4. STFT 변환 (Processor에서 (2, Freq, Time) 텐서 반환함)
        mix_mag, _ = self.processor.audio_to_stft(mix)
        
        sources_mags = []
        for i in range(4):
            mag, _ = self.processor.audio_to_stft(sources[i])
            sources_mags.append(mag)
            
        # Target Shape: (4, 2, Freq, Time)
        sources_mag = torch.stack(sources_mags)
        
        # Return: 
        # Input: (2, Freq, Time)
        # Target: (4, 2, Freq, Time)
        return mix_mag, sources_mag