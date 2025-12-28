import librosa
import numpy as np
import torch

class AudioProcessor:
    def __init__(self, sr=44100, n_fft=2048, hop_length=512):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        # 모델 입력 크기를 맞추기 위해 주파수 빈을 1024로 자름
        self.n_bins = n_fft // 2 

    def load_audio(self, path, duration=None): # [수정] duration 인자 추가
        """
        스테레오(2ch)로 오디오 로드
        Return: (2, Samples) 형태의 torch.Tensor
        """
        # [수정] duration을 지원하도록 librosa.load 호출부 변경
        # mono=False로 설정하여 스테레오 유지
        audio, _ = librosa.load(path, sr=self.sr, mono=False, duration=duration)
        
        # 만약 원본이 모노(1ch)라면 (2, Samples)로 복제
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=0)
            
        # [추가] 훈련 시 duration에 따른 샘플 수 강제 고정 (길이가 짧은 파일 대비)
        if duration is not None:
            target_samples = int(self.sr * duration)
            if audio.shape[1] < target_samples:
                audio = np.pad(audio, ((0, 0), (0, target_samples - audio.shape[1])), mode='constant')
            elif audio.shape[1] > target_samples:
                audio = audio[:, :target_samples]
            
        return torch.from_numpy(audio).float()

    def audio_to_stft(self, audio):
        """
        스테레오 오디오 -> Magnitude, Phase 분리
        Input: (2, Samples) Tensor 또는 Numpy
        Return: (Magnitude Tensor, Phase Numpy)
        """
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()

        mags = []
        phases = []

        # 채널별(Left, Right)로 STFT 수행
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
            
        mag_tensor = torch.from_numpy(np.stack(mags, axis=0)).float()
        phase_arr = np.stack(phases, axis=0)
        
        return mag_tensor, phase_arr

    def stft_to_audio(self, mag, phase):
        # ... (기존 stft_to_audio 로직과 동일하여 생략, 기존 코드 그대로 사용하세요)
        if isinstance(mag, torch.Tensor):
            mag = mag.detach().cpu().numpy()
            
        if mag.ndim == 4: 
            mag = mag[0]
            
        channels = []
        for ch in range(2):
            curr_mag = mag[ch]
            curr_phase = phase[ch]
            curr_mag = np.expm1(curr_mag)
            curr_mag = np.pad(curr_mag, ((0, 1), (0, 0)), mode='constant')
            curr_phase = np.pad(curr_phase, ((0, 1), (0, 0)), mode='constant')
            stft = curr_mag * np.exp(1j * curr_phase)
            audio_ch = librosa.istft(stft, hop_length=self.hop_length)
            channels.append(audio_ch)
            
        return np.stack(channels, axis=0)