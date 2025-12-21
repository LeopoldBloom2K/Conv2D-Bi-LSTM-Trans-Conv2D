import mlflow
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def calculate_sdr(reference, estimate):
    """
    간이 SDR (Signal-to-Distortion Ratio) 계산 함수
    수식이 복잡한 bss_eval 대신 Numpy로 빠르게 계산
    Higher is better.
    """
    # 길이 맞추기 (짧은 쪽에 맞춤)
    min_len = min(len(reference), len(estimate))
    reference = reference[:min_len]
    estimate = estimate[:min_len]

    # 노이즈(오차) 계산
    noise = reference - estimate
    
    # 에너지 계산 (작은 값 더해 0 나누기 방지)
    s_true = np.sum(reference ** 2) + 1e-7
    s_noise = np.sum(noise ** 2) + 1e-7
    
    sdr = 10 * np.log10(s_true / s_noise)
    return sdr

def compare_models(ref_path, model_a_path, model_b_path, exp_name="Model_Comparison"):
    # 1. 오디오 로드
    print("Loading audio files...")
    y_ref, sr = librosa.load(ref_path, sr=22050)
    y_a, _ = librosa.load(model_a_path, sr=22050)
    y_b, _ = librosa.load(model_b_path, sr=22050)

    # 2. 정확도(SDR) 계산
    sdr_a = calculate_sdr(y_ref, y_a)
    sdr_b = calculate_sdr(y_ref, y_b)

    print(f"Desktop (Model A) SDR: {sdr_a:.2f} dB")
    print(f"Laptop  (Model B) SDR: {sdr_b:.2f} dB")

    # 승자 판별
    if sdr_a > sdr_b:
        winner = "Desktop (Model A)"
        best_sdr = sdr_a
    else:
        winner = "Laptop (Model B)"
        best_sdr = sdr_b

    print(f"🏆 Winner: {winner}")

    # 3. 스펙트로그램 시각화 및 저장
    plt.figure(figsize=(12, 12))
    
    # Reference
    plt.subplot(3, 1, 1)
    D_ref = librosa.amplitude_to_db(np.abs(librosa.stft(y_ref)), ref=np.max)
    librosa.display.specshow(D_ref, sr=sr, x_axis='time', y_axis='hz')
    plt.title('Ground Truth (Reference)')
    
    # Model A
    plt.subplot(3, 1, 2)
    D_a = librosa.amplitude_to_db(np.abs(librosa.stft(y_a)), ref=np.max)
    librosa.display.specshow(D_a, sr=sr, x_axis='time', y_axis='hz')
    plt.title(f'Desktop Model (SDR: {sdr_a:.2f} dB)')
    
    # Model B
    plt.subplot(3, 1, 3)
    D_b = librosa.amplitude_to_db(np.abs(librosa.stft(y_b)), ref=np.max)
    librosa.display.specshow(D_b, sr=sr, x_axis='time', y_axis='hz')
    plt.title(f'Laptop Model (SDR: {sdr_b:.2f} dB)')
    
    plt.tight_layout()
    plot_path = "comparison_result.png"
    plt.savefig(plot_path)
    plt.close()

    # 4. MLflow 기록
    mlflow.set_experiment(exp_name)
    
    with mlflow.start_run():
        # 파라미터 기록 (파일 경로)
        mlflow.log_param("ref_file", os.path.basename(ref_path))
        mlflow.log_param("model_a_file", os.path.basename(model_a_path))
        mlflow.log_param("model_b_file", os.path.basename(model_b_path))
        
        # 메트릭 기록 (점수)
        mlflow.log_metric("SDR_Desktop", sdr_a)
        mlflow.log_metric("SDR_Laptop", sdr_b)
        mlflow.log_metric("SDR_Diff", abs(sdr_a - sdr_b))
        
        # 태그 기록 (승자 표시)
        mlflow.set_tag("Winner", winner)
        
        # 아티팩트 저장 (이미지 및 오디오 파일)
        mlflow.log_artifact(plot_path)
        # 필요하다면 결과 오디오도 업로드 가능 (용량 주의)
        # mlflow.log_artifact(model_a_path)
        # mlflow.log_artifact(model_b_path)
        
        print("✅ MLflow logging complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, required=True, help='Path to Ground Truth (Original Vocals)')
    parser.add_argument('--a', type=str, required=True, help='Path to Result A (Desktop)')
    parser.add_argument('--b', type=str, required=True, help='Path to Result B (Laptop)')
    
    args = parser.parse_args()
    
    compare_models(args.ref, args.a, args.b)