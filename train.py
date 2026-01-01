import torch
from torch.utils.data import DataLoader, ConcatDataset
import os
import datetime
import glob

from options import get_args
from trainer import Trainer
from models.crnn_separator import CRNN_Separator
from utils.audio_processor import AudioProcessor
from utils.dataset import RemixingDataset

def load_pretrained_weights(model, checkpoint_path, device):
    """
    기존 모델과 현재 모델의 구조(Shape)가 달라도 
    일치하는 가중치만 똑똑하게 가져오는 함수
    (예: 1-stem 모델 가중치를 4-stem 모델로 이식할 때 사용)
    """
    print(f"♻️ Loading weights from: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        
        model_dict = model.state_dict()
        
        # 1. 모양이 일치하는 레이어만 필터링
        pretrained_dict = {
            k: v for k, v in state_dict.items() 
            if k in model_dict and v.shape == model_dict[k].shape
        }
        
        # 2. 로드되지 않는 레이어(모양이 다른 레이어) 확인
        ignored_layers = [k for k in model_dict.keys() if k not in pretrained_dict]
        
        # 3. 가중치 업데이트
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
        print(f"   ✅ 성공적으로 로드된 레이어: {len(pretrained_dict)}개")
        if len(ignored_layers) > 0:
            print(f"   ⚠️ 구조가 달라 초기화된 레이어 (재학습 필요): {len(ignored_layers)}개")
            print(f"      -> 예: {ignored_layers[:3]} ...")
            
    except Exception as e:
        print(f"   ❌ 로드 실패: {e}")

def main():
    start_time = datetime.datetime.now()
    timestamp = start_time.strftime('%Y%m%d_%H%M%S')
    
    args = get_args()
    
    # Fine-tuning 시 실험 이름 자동 변경
    if args.pretrained_path:
        args.exp_name = f"Finetune_{args.exp_name}_{timestamp}"
    else:
        args.exp_name = f"{args.exp_name}_{timestamp}"
    
    print("="*50)
    print(f"⏰ 학습 시작: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📝 실험 이름: {args.exp_name}")
    print(f"🎯 목표: 4-Stem Separation (Vocals, Drums, Bass, Other)")
    print("="*50)

    # 체크포인트 폴더 생성
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = AudioProcessor(sr=args.sr, n_fft=args.n_fft, hop_length=args.hop_length)
    
    # ---------------------------------------------------------
    # 1. 데이터셋 구성 (여러 데이터셋 병합)
    # ---------------------------------------------------------
    # 예: --train_dir에 "data/musdb18,data/moises,data/slakh" 처럼 콤마로 구분해서 넣거나
    # 아래 리스트에 직접 경로를 추가하세요.
    
    # [사용자 수정 영역] 사용할 데이터셋 경로 리스트
    # args.train_dir가 콤마(,)로 구분되어 들어온다고 가정하거나 리스트 직접 작성
    if ',' in args.train_dir:
        train_dirs = args.train_dir.split(',')
    else:
        train_dirs = [args.train_dir] 
        # 필요하다면 여기에 강제로 추가 가능: 
        # train_dirs = ['/path/to/musdb', '/path/to/moises', '/path/to/slakh']

    print(f"📂 학습 데이터셋 경로 병합 중... ({len(train_dirs)}개 소스)")
    
    train_datasets = []
    for d_path in train_dirs:
        d_path = d_path.strip()
        if os.path.exists(d_path):
            print(f"   -> 추가: {d_path}")
            # Moises/Slakh 등 데이터 양이 많으므로 remix_prob를 0.5~0.8로 적극 활용 추천
            ds = RemixingDataset(d_path, processor, duration=3.0, remix_prob=0.5) 
            train_datasets.append(ds)
        else:
            print(f"   ⚠️ 경고: 경로를 찾을 수 없음 - {d_path}")

    if not train_datasets:
        raise ValueError("❌ 유효한 학습 데이터 경로가 없습니다!")

    # 데이터셋 병합 (ConcatDataset)
    combined_train_dataset = ConcatDataset(train_datasets)
    
    # 검증 데이터셋 (기존 유지)
    val_dataset = RemixingDataset(args.val_dir, processor, duration=3.0, remix_prob=0.0)
    
    train_loader = DataLoader(
        combined_train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    
    print(f"📊 총 학습 샘플 수: {len(combined_train_dataset)}")
    
    # ---------------------------------------------------------
    # 2. 모델 초기화 (4-Stem 타겟)
    # ---------------------------------------------------------
    model = CRNN_Separator(
        input_channels=2, 
        n_bins=args.n_fft // 2, 
        num_stems=4,          # [중요] Demucs 대체용이므로 4로 고정
        hidden_size=args.hidden_size, 
        num_layers=args.num_layers
    ).to(device)
    
    # ---------------------------------------------------------
    # 3. Pre-trained Weights 로드 (Transfer Learning)
    # ---------------------------------------------------------
    if args.pretrained_path and os.path.exists(args.pretrained_path):
        load_pretrained_weights(model, args.pretrained_path, device)
        
        # [Fine-tuning 팁] LR 자동 조절 제안
        if args.lr > 0.0005:
            print("\n🚨 [주의] Fine-tuning 시에는 Learning Rate를 낮추는 것이 좋습니다.")
            print(f"   현재 LR: {args.lr} -> 권장 LR: 0.0001 ~ 0.0002")
    else:
        print("\n🚀 Pre-trained 모델 없이 처음부터 학습합니다 (Scratch Training).")

    # ---------------------------------------------------------
    # 4. 학습 시작
    # ---------------------------------------------------------
    trainer = Trainer(model, train_loader, val_loader, args)
    trainer.fit()

    end_time = datetime.datetime.now()
    print(f"⏳ 전체 소요 시간: {end_time - start_time}")

if __name__ == '__main__':
    main()