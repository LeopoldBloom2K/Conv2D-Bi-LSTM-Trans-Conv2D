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
        """
        스테레오(2ch)로 오디오 로드
        Return: (2, Samples) 형태의 Numpy Array
        """
        # mono=False로 설정하여 스테레오 유지
        audio, _ = librosa.load(path, sr=self.sr, mono=False)
        
        # 만약 원본이 모노(1ch)라면 (2, Samples)로 복제
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=0)
            
        return audio

    def audio_to_stft(self, audio):
        """
        스테레오 오디오 -> Magnitude, Phase 분리
        Input: (2, Samples)
        Return: (Magnitude Tensor, Phase Numpy)
        - Magnitude Shape: (2, Freq, Time)
        """
        mags = []
        phases = []

        # 채널별(Left, Right)로 루프를 돌며 STFT 수행
        for ch in range(2):
            stft = librosa.stft(audio[ch], n_fft=self.n_fft, hop_length=self.hop_length)
            
            # 마지막 주파수 빈 버리기 (1025 -> 1024)
            stft = stft[:-1, :]
            
            mag = np.abs(stft)
            phase = np.angle(stft)
            
            # Log-scale 변환
            mag = np.log1p(mag)
            
            mags.append(mag)
            phases.append(phase)
            
        # (2, Freq, Time) 형태로 쌓기
        mag_tensor = torch.from_numpy(np.stack(mags, axis=0)).float()
        phase_arr = np.stack(phases, axis=0)
        
        return mag_tensor, phase_arr

    def stft_to_audio(self, mag, phase):
        """
        Magnitude + Phase -> 스테레오 오디오 복원
        Input Mag: (2, Freq, Time) or (Batch, 2, Freq, Time)
        """
        if isinstance(mag, torch.Tensor):
            mag = mag.detach().cpu().numpy()
            
        # 배치 차원이 있다면 첫 번째 배치만 가져옴 (추론 시)
        if mag.ndim == 4: 
            mag = mag[0]
            
        channels = []
        for ch in range(2):
            # 채널별 데이터 가져오기
            curr_mag = mag[ch]
            curr_phase = phase[ch]

            # Log-scale 역변환
            curr_mag = np.expm1(curr_mag)
            
            # 버렸던 마지막 빈(bin) 복구
            curr_mag = np.pad(curr_mag, ((0, 1), (0, 0)), mode='constant')
            curr_phase = np.pad(curr_phase, ((0, 1), (0, 0)), mode='constant')
            
            # ISTFT
            stft = curr_mag * np.exp(1j * curr_phase)
            audio_ch = librosa.istft(stft, hop_length=self.hop_length)
            channels.append(audio_ch)
            
        # (2, Samples) 형태로 결합
        return np.stack(channels, axis=0)