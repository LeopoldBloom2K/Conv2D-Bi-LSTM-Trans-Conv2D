# CRNN 음악 소스 분리 프로젝트

이 프로젝트는 CRNN (Convolutional Recurrent Neural Network) 아키텍처를 기반으로 한 경량화된 음악 소스 분리(Music Source Separation) 모델입니다. 입력된 음원(Mixture)에서 보컬, 드럼 등 특정 악기 트랙을 분리해냅니다.

## ✨ 핵심 기능 (Key Features)

* **Hybrid Architecture** : CNN(공간적 특징) + Bi-GRU(시간적 흐름/템포) + U-Net(Skip Connection) 결합 구조

* **Optimization** : 22.05kHz 샘플링, FP16(AMP) 연산 적용으로 학습 속도 향상 및 메모리 점유율 최소화

* **Data Augmentation** : 학습 시 실시간 트랙 믹싱(On-the-fly Remixing)을 통한 데이터 증강 및 과적합 방지

* **Smart Training** : Validation Loss 기반의 Early Stopping 및 Best Model 자동 저장 지원

## 📂 데이터셋 (Datasets)

본 모델 학습을 위해 다음과 같은 **Mix(혼합음)**, **Stems(개별 악기 트랙)** 가 포함된 데이터셋을 사용함.

| 데이터셋 | 특징 | 다운로드 링크 | 비고 |
| :--- | :--- | :--- | :--- |
| MUSDB18-HQ 업계 표준 | 고음질(WAV), 150곡 | SigSep/MUSDB18 | 입문 및 메인 학습용 |
| MoisesDB | 대규모 데이터, 다양한 장르, 240곡 | Moises-AI/MoisesDB | 범용성 확보용 |
| Slakh2100 | MIDI 기반 가상 악기, 노이즈 없음, 2100곡 | Slakh Dataset | 사전 학습(Pre-training)용 |

> **⚠️주의** : 다운로드 후 `data/train/곡이름/` 폴더 내에 `mixture.wav`와 `vocals.wav`(타겟 악기)가 위치하도록 폴더 구조 정리 필요.

## 프로젝트 구조

```
.
├── data/
│   └── train/              # 학습 데이터 (MUSDB18 등)
│       ├── Song_Title_1/
│       │   ├── mixture.wav
│       │   └── vocals.wav  # (또는 drums.wav, bass.wav ...)
│       └── ...
├── checkpoints/            # 학습된 모델과 체크포인트 저장소
├── models/
│   └── crnn_separator.py   # 모델 정의 (Conv + Bi-GRU + U-Net)
├── utils/
│   ├── audio_processor.py  # STFT/Phase 처리 및 오디오 로딩
│   ├── dataset.py          # Remixing을 포함한 데이터셋 로더
│   └── early_stopping.py   # 학습 조기 종료 및 Best Model 저장 로직
├── train.py                # 학습 실행 스크립트
├── inference.py            # 음원 분리(추론) 스크립트
└── requirements.txt        # 의존성 패키지 목록
```

## 설치 방법

```bash
pip install -r requirements.txt
```

## 데이터 준비

학습을 위해서는 **Mix(섞인 곡)**와 Target(분리할 악기) 음원 쌍이 필요합니다.</br>
`data/train/` 폴더 아래에 곡 별로 폴더를 만들고, 그 안에 `mixture.wav`와 `vocals.wav`(타겟 악기)를 넣어주세요.

# 학습

```bash
# 기본 학습 (Vocals 분리)
python train.py --data_dir ./data/train --target vocals

# 옵션 변경 예시 (드럼 분리, 배치 사이즈 32, 에폭 1000)
python train.py --data_dir ./data/train --target drums --batch_size 32 --epochs 1000
```

> **Tip** : epochs를 크게 설정해도 Early Stopping 기능이 있어 성능 향상이 멈추면 자동으로 학습을 종료하고 가장 좋은 모델을 저장합니다.

## 학습 재개 (Resume/ Fine-tuning)

* 중단된 지점부터 다시 학습하거나, 사전 학습된 모델을 불러오려면 `--resume_from`을 사용하세요.

```bash
python train.py --data_dir ./data/train --resume_from ./checkpoints/best_model.pth
```

## 추론 (Inference)

* 학습된 모델(.pth)을 사용하여 새로운 노래를 분리합니다.

```bash
python inference.py \
    --input_path ./my_song.mp3 \
    --output_path ./separated_vocals.wav \
    --model_path ./checkpoints/best_model.pth
```

## 모델 아키텍처

* **Encoder**: 4-layer CNN (주파수 차원 압축)
* **Bottleneck**: Bi-GRU (3 layers) - LSTM 대비 연산량 30% 절감
* **Decoder**: Transposed CNN (Skip Connection 적용으로 디테일 복원)
* **Skip Connection**: 인코더의 특징맵을 디코더로 전달하여 음질 손실 최소화

## Audio Config (Optimized)

가성비와 학습 속도를 위해 최적화된 설정입니다.

| Parameter | Value | Description |
| :--- | :--- | :--- |
| Sample Rate | 22,050 Hz | 11kHz 이하 정보 집중 (데이터 크기 50% 절감) |
| N_FFT | 1024 | 주파수 해상도 |
| Hop Length | 256 | 시간 해상도 |
| Input Channels | 1 | Mono Audio |

## Traing config

| Parameter | Value | Description |
| :--- | :--- | :--- |
| Optimizer | Adam | Weight Decay 1e-5 적용 |
| Loss Function | L1 Loss | 오디오 복원에 유리 |
| Batch Size | 32 | AMP 적용으로 넉넉하게 설정 가능 |
| Precision | Mixed (FP16) | GPU 메모리 절약 및 연산 가속 |
| Early Stopping | Patience=15 | 과적합 방지 |
