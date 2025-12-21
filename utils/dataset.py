import torch
from torch.utils.data import Dataset
import os
import random
import numpy as np

class RemixingDataset(Dataset):
    def __init__(self, data_dir, processor, target_name='vocals', duration=3.0, remix_prob=0.5):
        """
        remix_prob: 리믹싱을 할 확률 (0.5면 50%는 원곡, 50%는 리믹스)
        """
        self.data_dir = data_dir
        self.processor = processor
        self.target_name = target_name
        self.duration = duration
        self.remix_prob = remix_prob
        
        # 1. 모든 악기 파일의 경로를 따로따로 저장 (Dictionary)
        self.sources = {'vocals': [], 'drums': [], 'bass': [], 'other': []}
        self.song_folders = [] # 원곡 유지를 위한 폴더 리스트
        
        self._scan_files()
        
    def _scan_files(self):
        # 폴더를 돌면서 각 악기 파일 경로 수집
        print(f"Scanning files in {self.data_dir}...")
        for root, dirs, files in os.walk(self.data_dir):
            # 현재 폴더에 4개 악기가 모두 있는지 확인 (MUSDB18 구조)
            # 파일명이 vocals.wav, drums.wav 등이어야 함
            has_all = all(f"{stem}.wav" in files for stem in self.sources.keys())
            
            if has_all:
                self.song_folders.append(root) # 원곡 폴더 저장
                for stem in self.sources.keys():
                    self.sources[stem].append(os.path.join(root, f"{stem}.wav"))
                    
        print(f"Dataset Loaded: {len(self.song_folders)} songs found.")

    def __len__(self):
        return len(self.song_folders)

    def load_random_chunk(self, path):
        # 오디오 로드 및 랜덤 크롭
        audio = self.processor.load_audio(path)
        
        # 필요한 샘플 수 계산
        req_samples = int(self.processor.sr * self.duration)
        
        if len(audio) < req_samples:
            # 짧으면 0으로 채움 (Padding)
            return np.pad(audio, (0, req_samples - len(audio)))
        else:
            # 길면 랜덤하게 자름
            start = random.randint(0, len(audio) - req_samples)
            return audio[start : start + req_samples]

    def __getitem__(self, idx):
        # --- A. 리믹싱 할까? 말까? 결정 ---
        do_remix = random.random() < self.remix_prob
        
        if not do_remix:
            # [Option 1] 원곡 그대로 사용 (기존 방식)
            folder = self.song_folders[idx]
            
            mix_path = os.path.join(folder, "mixture.wav")
            target_path = os.path.join(folder, f"{self.target_name}.wav")
            
            # 원곡도 랜덤 크롭 적용
            mix_audio = self.load_random_chunk(mix_path)
            target_audio = self.load_random_chunk(target_path)
            
        else:
            # [Option 2] 리믹싱 (Remixing) - 과적합 방지의 핵심!
            sources = {}
            
            # 1. 타겟 악기 (예: vocals)는 현재 인덱스(idx)의 것을 사용
            target_path = self.sources[self.target_name][idx]
            target_audio = self.load_random_chunk(target_path)
            sources[self.target_name] = target_audio
            
            # 2. 나머지 악기 (반주)는 랜덤한 다른 노래에서 가져옴
            mix_audio = target_audio.copy() # 시작은 타겟 소리
            
            for stem in ['drums', 'bass', 'other']:
                if stem == self.target_name: continue # 타겟은 중복 로드 X
                
                # 랜덤하게 다른 노래 파일 선택
                random_path = random.choice(self.sources[stem])
                stem_audio = self.load_random_chunk(random_path)
                
                # 3. 합치기 (Gain 조절로 클리핑 방지 추천)
                gain = random.uniform(0.7, 1.3) # 볼륨을 살짝 랜덤하게
                mix_audio += (stem_audio * gain)
        
        # 클리핑 방지 (최대 볼륨이 1.0 넘지 않게)
        max_val = np.max(np.abs(mix_audio))
        if max_val > 1.0:
            mix_audio /= max_val
            target_audio /= max_val 

        # --- B. STFT 변환 및 반환 ---
        mix_mag, _ = self.processor.audio_to_stft(mix_audio)
        target_mag, _ = self.processor.audio_to_stft(target_audio)
        
        return mix_mag, target_mag