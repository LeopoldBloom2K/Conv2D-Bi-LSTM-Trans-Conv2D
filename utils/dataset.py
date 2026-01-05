import torch
from torch.utils.data import Dataset
import os
import random
import numpy as np
import librosa
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class RemixingDataset(Dataset):
    def __init__(self, data_dir, processor, duration=3.0, remix_prob=0.5, target_stems=None):
        self.data_dir = data_dir
        self.processor = processor
        self.duration = duration
        self.remix_prob = remix_prob
        
        if target_stems is None:
            self.target_stems = ['vocals', 'drums', 'bass', 'other']
        else:
            self.target_stems = target_stems

        self.song_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) 
                          if os.path.isdir(os.path.join(data_dir, d))]
        
        self.n_fft = getattr(processor, 'n_fft', 2048)
        self.hop_length = getattr(processor, 'hop_length', 512)
        self.sr = getattr(processor, 'sr', 44100)

    def load_and_process(self, idx):
        song_dir = self.song_dirs[idx]
        
        # 1. Measure length for random section
        check_file = os.path.join(song_dir, f"{self.target_stems[0]}.wav")
        if not os.path.exists(check_file):
             check_file = os.path.join(song_dir, "mixture.wav")
        
        start_offset = 0.0
        try:
            if os.path.exists(check_file):
                total_duration = librosa.get_duration(path=check_file)
                if total_duration > self.duration:
                    start_offset = random.uniform(0, total_duration - self.duration)
        except Exception:
            start_offset = 0.0

        # 2. Load stems and STFT
        ref_shape = None
        final_stems = []

        for name in self.target_stems:
            path = os.path.join(song_dir, f"{name}.wav")
            mag = None
            
            if os.path.exists(path):
                try:
                    audio, _ = librosa.load(
                        path, 
                        sr=self.sr, 
                        offset=start_offset, 
                        duration=self.duration, 
                        mono=True 
                    )
                    
                    spec = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
                    mag_numpy = np.abs(spec)

                    # -----------------------------------------------------------
                    # [CRITICAL FIX] Log Transformation (Linear -> Log Scale)
                    # Compresses large linear amplitude values to prevent model explosion.
                    # -----------------------------------------------------------
                    mag_numpy = np.log1p(mag_numpy)
                    
                    # [CRITICAL FIX 1] Slice 513 -> 512 (Remove Nyquist Bin)
                    # Matches model config (n_bins = n_fft // 2)
                    if mag_numpy.shape[0] == (self.n_fft // 2) + 1:
                        mag_numpy = mag_numpy[:-1, :]
                    
                    mag = torch.from_numpy(mag_numpy).float()
                    
                    if len(mag.shape) == 2:
                        mag = mag.unsqueeze(0) 

                    if mag.shape[0] == 1:
                        mag = mag.repeat(2, 1, 1)

                    if ref_shape is None:
                        ref_shape = mag.shape
                        
                except Exception as e:
                    print(f"❌ Processing failed ({path}): {e}")
                    mag = None
            
            final_stems.append(mag)

        # 3. Data Packaging
        processed_stems = []
        
        # [CRITICAL FIX 2] Set reference shape to 512 bins
        if ref_shape is None:
             n_bins = self.n_fft // 2  # +1 removed (512)
             n_frames = int(self.sr * self.duration / self.hop_length) + 1
             ref_shape = (2, n_bins, n_frames)

        for mag in final_stems:
            if mag is not None:
                if mag.shape[-1] != ref_shape[-1]:
                    min_len = min(mag.shape[-1], ref_shape[-1])
                    mag = mag[..., :min_len]
                processed_stems.append(mag)
            else:
                zero_tensor = torch.zeros(ref_shape, dtype=torch.float32)
                processed_stems.append(zero_tensor)

        targets = torch.stack(processed_stems)
        
        min_time = min([t.shape[-1] for t in targets])
        targets = targets[..., :min_time]
        
        mix = torch.sum(targets, dim=0)
        return mix, targets

    def __getitem__(self, idx):
        mix, targets = self.load_and_process(idx)

        # Augmentation
        if self.remix_prob > 0:
            if random.random() < 0.2:
                targets[0] = torch.zeros_like(targets[0])

            for i in range(targets.shape[0]):
                gain = random.uniform(0.7, 1.3)
                targets[i] = targets[i] * gain
            
            if random.random() > 0.5:
                mix = torch.flip(mix, dims=[0])
                targets = torch.flip(targets, dims=[1])

            mix = torch.sum(targets, dim=0)
            max_val = mix.abs().max()
            if max_val > 1.0:
                scale = 0.99 / max_val
                mix = mix * scale
                targets = targets * scale

        return mix, targets

    def __len__(self):
        return len(self.song_dirs)