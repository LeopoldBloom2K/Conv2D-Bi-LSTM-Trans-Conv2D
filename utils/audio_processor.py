import librosa
import numpy as np
import torch

class AudioProcessor:
    def __init__(self, sr=44100, n_fft=2048, hop_length=512):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        # 모델 입력 크기를 맞추기 위해 주파수 빈을 1024로 자름 (원래 n_fft//2 + 1 = 1025)
        self.n_bins = n_fft // 2 

    def load_audio(self, path):
        audio, _ = librosa.load(path, sr=self.sr, mono=True)
        return audio

    def audio_to_stft(self, audio):
        """
        오디오 -> Magnitude(크기), Phase(위상) 분리
        Return: (Magnitude Tensor, Phase Numpy)
        """
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        
        # 마지막 주파수 빈 하나를 버려서 1024개로 맞춤 (모델의 MaxPool 크기 호환성 위함)
        stft = stft[:-1, :] 
        
        mag = np.abs(stft)
        phase = np.angle(stft)
        
        # Log-scale 변환 (학습 안정성)
        mag = np.log1p(mag)
        
        return torch.FloatTensor(mag).unsqueeze(0), phase

    def stft_to_audio(self, mag, phase):
        """
        Magnitude + Phase -> 오디오 복원
        """
        if isinstance(mag, torch.Tensor):
            mag = mag.detach().cpu().numpy().squeeze()
            
        # Log-scale 역변환
        mag = np.expm1(mag)
        
        # 버렸던 마지막 빈(bin) 0으로 채워서 복구
        mag = np.pad(mag, ((0, 1), (0, 0)), mode='constant')
        phase = np.pad(phase, ((0, 1), (0, 0)), mode='constant')
        
        # 복소수 재구성
        stft = mag * np.exp(1j * phase)
        audio = librosa.istft(stft, hop_length=self.hop_length)
        return audio