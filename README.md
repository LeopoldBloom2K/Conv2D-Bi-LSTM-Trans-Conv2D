# CRNN 음악 소스 분리 프로젝트

## 프로젝트 구조
```
.
├── data/
│   ├── raw/              # 원본 오디오 데이터
│   └── processed/        # 전처리된 데이터
├── models/
│   └── crnn_separator.py # CRNN 모델 정의
├── utils/
│   ├── audio_processor.py # 오디오 전처리
│   └── dataset.py        # 데이터셋 클래스
├── train.py              # 학습 스크립트
├── inference.py          # 추론 스크립트
└── requirements.txt      # 의존성 패키지
```

## 설치 방법
```bash
pip install -r requirements.txt
```

## 데이터 준비
`data/raw/` 디렉토리에 오디오 파일을 추가하세요.

## 학습
```bash
python train.py
```

## 추론
```bash
python inference.py
```

## 모델 아키텍처
- **Encoder**: Conv2D로 주파수 특징 추출
- **Bottleneck**: Bi-LSTM으로 시간적 흐름 학습
- **Decoder**: TransposedConv2D로 마스크 생성
- **Skip Connection**: U-Net 스타일로 디테일 보존

```
model:
  input_channels: 1
  hidden_size: 128
  freq_bins: 2048
```
# 오디오 설정
```
audio:
  sample_rate: 44100
  n_fft: 4096
  hop_length: 1024
  chunk_duration: 4.0
  overlap: 0.5
```
# 학습 설정
```
training:
  batch_size: 8
  learning_rate: 0.001
  num_epochs: 100
  save_interval: 10
```
# 경로 설정
```
paths:
  train_data: data/raw/
  processed_data: data/processed/
  checkpoints: checkpoints/
  output: output/
```