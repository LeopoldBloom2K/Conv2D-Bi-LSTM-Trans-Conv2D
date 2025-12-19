import torch
from torch.utils.data import Dataset
import os
import glob

class MusicSeparationDataset(Dataset):
    def __init__(self, data_dir, processor, target_name='vocals', duration=3.0):
        """
        data_dir 구조: data_dir/song_name/mixture.wav & vocals.wav
        """
        self.data_dir = data_dir
        self.processor = processor
        self.target_name = target_name
        self.duration = duration
        self.file_pairs = self._find_pairs()

    def _find_pairs(self):
        pairs = []
        # 하위 폴더 검색
        songs = [f.path for f in os.scandir(self.data_dir) if f.is_dir()]
        
        for song_path in songs:
            mix_path = os.path.join(song_path, 'mixture.wav')
            target_path = os.path.join(song_path, f'{self.target_name}.wav')
            
            if os.path.exists(mix_path) and os.path.exists(target_path):
                pairs.append((mix_path, target_path))
        
        print(f"Found {len(pairs)} song pairs for training.")
        return pairs

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        mix_path, target_path = self.file_pairs[idx]
        
        # 오디오 로드 (랜덤 크롭 로직은 생략하고, 간단히 앞부분 로드 예시)
        # 실제 학습시는 전체 곡을 랜덤하게 자르는 로직(Chunking)이 필요함
        mix_audio = self.processor.load_audio(mix_path)
        target_audio = self.processor.load_audio(target_path)
        
        # 길이 맞추기 (짧은 쪽에 맞춤)
        min_len = min(len(mix_audio), len(target_audio))
        # 학습용으로 고정된 길이(예: 3초)만 자르기
        max_samples = int(self.processor.sr * self.duration)
        if min_len > max_samples:
            start = np.random.randint(0, min_len - max_samples)
            mix_audio = mix_audio[start : start + max_samples]
            target_audio = target_audio[start : start + max_samples]
        else:
            mix_audio = mix_audio[:min_len]
            target_audio = target_audio[:min_len]

        # STFT 변환 (Phase는 학습 때 필요 없음)
        mix_mag, _ = self.processor.audio_to_stft(mix_audio)
        target_mag, _ = self.processor.audio_to_stft(target_audio)
        
        return mix_mag, target_mag