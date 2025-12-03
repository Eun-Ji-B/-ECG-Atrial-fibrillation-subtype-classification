##################
# 파일 읽어오는 코드
import wfdb
import numpy as np
import pandas as pd
import os
import time  # 시간 측정용 라이브러리
import psutil
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


# 파일 처리 함수
def process_ecg_files(file_path, file_name):
    # 원본 파일 경로에서 읽기
    dat_file = os.path.join(file_path, file_name + '.dat')
    hea_file = os.path.join(file_path, file_name + '.hea')
    atr_file = os.path.join(file_path, file_name + '.atr')

    # ECG 신호 읽기
    record = wfdb.rdrecord(dat_file[:-4])  # `sampto` 값을 제거하여 전체 신호를 읽기
    ecg_signal = record.p_signal[:, 1]  # npy 신호

    # 헤더 정보 읽기
    header = wfdb.rdheader(hea_file[:-4])
    sampling_rate = header.fs    

    label = ""
    with open(hea_file, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                label = line.lstrip("#").strip()
                break

    return ecg_signal, sampling_rate, label

##################

##################
# 전처리 코드 To CQT
#------대역통과 필터------
from scipy.signal import butter, filtfilt

def design_bandpass(low_hz, high_hz, order, fs = 200.0):
    nyq = fs * 0.5
    high = min(high_hz, nyq * 0.98)
    low  = max(low_hz, 0.001)
    if low >= high:  # 극단적 fs 방어
        low  = min(0.1 * nyq, nyq * 0.2)
        high = min(0.9 * nyq, nyq * 0.95)
    b, a = butter(order, [low/nyq, high/nyq], btype='bandpass')
    return b, a
#------10초 미만 제외------
def filter_one_record(sig):  # sig = ecg_signal
    n = sig.shape[0]
    fs = 200
    global MIN_SEC
    MIN_SEC = 10.0  
    
    if n < int(MIN_SEC * fs):
        return ("excluded_short")
    
    b, a = design_bandpass(0.5, 50.0, 1 ) #LOW_HZ, HIGH_HZ = 0.5, 50.0, ORDER = 1
    sig = filtfilt(b, a, sig, method="pad")
    
    return sig
#------10초 분할------
TARGET_FS = 200.0
SEGMENT_SECONDS = 10.0

# 10초 * 200Hz = 2000 샘플
SEGMENT_SAMPLES = int(TARGET_FS * SEGMENT_SECONDS)

# 사용할 리드 이름 (논문은 Lead II 사용 명시)
TARGET_LEAD = 'II'



def process_npy_record(sig):
    """
    단일 원본 NPY/CSV 파일을 로드, 전처리(논문 방식), 분할하여 저장합니다.
    """
    # 6. 10초(2000 샘플) 단위로 분할
    total_samples = len(sig)
    num_segments = total_samples // SEGMENT_SAMPLES

    if num_segments == 0:
        
        return 0
    
    # 7. 분할된 파일 저장 (NPY + CSV)
    made_segments = 0
    for i in range(num_segments):
        start = i * SEGMENT_SAMPLES
        end = start + SEGMENT_SAMPLES
        segment_data = sig[start:end] # 1D 배열
    return segment_data

#------CQT 변환------
INPUT_FS = 200.0
TARGET_FS = None
FMIN = 0.5
FMAX = 50.0
BINS_PER_OCT = 12
HOP_SEC = 0.01
USE_VQT = True
POWER_DB = True
MIN_DURATION_SEC = 9.9

import librosa
def compute_cqt_mag(x: np.ndarray, fs: float):
    hop_length = max(1, int(round(HOP_SEC * fs)))
    n_oct = np.log2(FMAX / FMIN)
    n_bins = int(np.ceil(n_oct * BINS_PER_OCT))
    if USE_VQT and hasattr(librosa, "vqt"): #VQT를 호출하였지만 파라미터 gamma=0.0으로 하면 CQT와 동일한 효과가 남
        C = librosa.vqt(x.astype(float), sr=fs, hop_length=hop_length, fmin=FMIN, n_bins=n_bins, bins_per_octave=BINS_PER_OCT, gamma=0.0, pad_mode="reflect")
    else:
        C = librosa.cqt(x.astype(float), sr=fs, hop_length=hop_length, fmin=FMIN, n_bins=n_bins, bins_per_octave=BINS_PER_OCT, pad_mode="reflect")
    S = np.abs(C) ** 2
    if POWER_DB: S_db = librosa.power_to_db(S, ref=np.max); return S_db, fs, hop_length, n_bins
    else: return S, fs, hop_length, n_bins
##################


##################
# TFlite 모델 추론
#import tensorflow as tf

from tflite_runtime.interpreter import Interpreter

MODEL_PATH = "./FP32_model1127.tflite"   # FP16 변환 가능

# ★ 추가: 모델 파일 크기 출력 (모델 크기 지표)
if os.path.exists(MODEL_PATH):
    model_size_bytes = os.path.getsize(MODEL_PATH)
    model_size_mb = model_size_bytes / (1024 ** 2)
    print(f"[INFO] TFLite 모델 크기: {model_size_mb:.2f} MB")
else:
    print(f"[WARN] 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")

interpreter = None
try:
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]["shape"]  # 예: [1, H, W, C]
    input_dtype = input_details[0]["dtype"]
    print(f"모델 입력 텐서 형태: {input_shape}")
    print(f"모델 입력 데이터 타입: {input_dtype}")
    print("quantization:", input_details[0]["quantization"])
except Exception as e:
    print(f"모델 로드 또는 텐서 할당 중 오류 발생: {e}")
    exit()


def run_inference_on_npy(cqt_2d):
    global interpreter, input_details, output_details, input_shape, input_dtype

    # 1) 2D CQT → float32 배열
    arr = np.array(cqt_2d, dtype=np.float32)   # (H, W)

    # 2) (H, W) → (H, W, 1)
    if arr.ndim == 2:
        arr = arr[..., np.newaxis]   # (H, W, 1)

    # 3) 모델이 요구하는 채널 수 확인
    #    input_shape: [1, H_in, W_in, C_in]
    target_h = input_shape[1]
    target_w = input_shape[2]
    target_c = input_shape[3] if len(input_shape) == 4 else 1

    # 3-1) 1채널을 3채널로 복제 (학습 코드와 동일: 1채널을 3채널로 repeat)
    if arr.shape[-1] == 1 and target_c == 3:
        arr = np.repeat(arr, repeats=3, axis=-1)  # (H, W, 3)

    # 3-2) H, W 크기 다르면 resize (가능하면 학습 때와 같은 크기로 CQT 만들기 권장)
    if arr.shape[0] != target_h or arr.shape[1] != target_w:
        import tensorflow as tf  # ← 원래 코드가 쓰고 있던 부분 (touch 안 함)
        arr = tf.image.resize(arr, (target_h, target_w)).numpy()

    # 4) 배치 차원 추가 → [1, H, W, C]
    input_data = np.expand_dims(arr, axis=0)      # (1, H, W, C)
    input_data = input_data.astype(np.float32)    # 항상 FP32 입력

    # 5) TFLite에 입력 세팅 + 추론
    interpreter.set_tensor(input_details[0]['index'], input_data)
    proc = psutil.Process(os.getpid())

    # ★ 추가: CPU 사용률 초기화 (구간 측정을 위해)
    _ = proc.cpu_percent(interval=1)

    invoke_start_time = time.time()
    before = proc.memory_info().rss
    invoke_start_used = psutil.virtual_memory().used
    
    interpreter.invoke()
    
    invoke_end_time = time.time()
    after = proc.memory_info().rss
    invoke_end_used = psutil.virtual_memory().used

    # ★ 추가: invoke 구간 CPU 사용률
    cpu_percent = proc.cpu_percent(interval=1)
    
    invoke_time = invoke_end_time - invoke_start_time
    ram_time = invoke_end_used - invoke_start_used
    ram_diff = after - before

    # 6) 출력 가져오기
    output_data = interpreter.get_tensor(output_details[0]["index"])  # (1, num_classes)
    probs = output_data[0]
    pred_class = int(np.argmax(probs))

    # ★ return 값 맨 끝에 cpu_percent 추가
    return (
        pred_class,
        probs,
        ram_time,
        invoke_time,
        invoke_start_time,
        invoke_end_time,
        invoke_start_used,
        invoke_end_used,
        before,
        after,
        ram_diff,
        cpu_percent,
    )


##################
def main():
    # 원본 데이터가 있는 디렉토리 경로 (입력 파일 경로)
    data_dir = './1127test'  # 'atr', 'dat', 'hea' 파일들이 있는 디렉토리
    ram = psutil.virtual_memory()
    # ▶ 학습 때 썼던 라벨 매핑과 맞춰야 함
    #   0: Non, 1: PAF, 2: PsAF
    label_to_idx = {
        "non atrial fibrillation": 0,
        "paroxysmal atrial fibrillation": 1,
        "persistent atrial fibrillation": 2,
    }
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    results = []   # 여기다가 파일별 결과를 다 쌓아둘 거예요
    total = 0
    correct = 0
    pred_idx = None
    probs = None
    ram_time = None
    invoke_time = None
    invoke_start_time = None
    invoke_end_time = None
    invoke_start_used = None
    invoke_end_used = None

    # ★ 추가: 처리량/CPU 평균 계산용 누적 변수
    total_invoke_time = 0.0
    total_cpu_percent = 0.0
    n_inference = 0

    # ★ 추가: 전체 추론 구간(전처리 + invoke 루프) 기준 측정 시작
    proc_total = psutil.Process(os.getpid())
    _ = proc_total.cpu_percent(interval=1)
    total_rss_before = proc_total.memory_info().rss
    total_start_time = time.time()
    peak_rss = 0 
    for fname in os.listdir(data_dir):
        # dat 파일만 대상
        if not fname.endswith(".dat"):
            continue

        file_id = fname[:-4]  # 예: "data_104_12"

        # 1) ECG + 헤더 라벨 읽기
        ecg_signal, sampling_rate, label_str = process_ecg_files(data_dir, file_id)

        # 1-1) 헤더 라벨이 우리가 정의한 label_to_idx에 없으면 스킵
        if label_str not in label_to_idx:
            print(f"[SKIP] 지원하지 않는 라벨: {file_id} -> '{label_str}'")
            continue

        true_idx = label_to_idx[label_str]

        # 2) 필터링 (길이 짧으면 제외)
        filter_sig = filter_one_record(ecg_signal)
        if isinstance(filter_sig, str) and filter_sig == "excluded_short":
            print(f"[SKIP] {file_id} : 길이 10초 미만으로 제외")
            continue

        # 3) 10초 세그먼트 하나 추출 (현재 코드는 마지막 10초만 사용)
        seg_10 = process_npy_record(filter_sig)

        # 4) CQT 계산 (2D CQT)
        s_db, fs_eff, hop_length, n_bins = compute_cqt_mag(seg_10, 200)

        # 5) TFLite 추론
        (
            pred_idx,
            probs,
            ram_time,
            invoke_time,
            invoke_start_time,
            invoke_end_time,
            invoke_start_used,
            invoke_end_used,
            before,
            after,
            ram_diff,
            cpu_percent,   # ★ 추가
        ) = run_inference_on_npy(s_db)

        # peak RSS 갱신
        if before > peak_rss:
            peak_rss = before
        if after > peak_rss:
            peak_rss = after

        
        # ★ 누적 (처리량/CPU 평균 계산용)
        total_invoke_time += invoke_time
        total_cpu_percent += cpu_percent
        n_inference += 1

        # 6) 결과 출력
        print(f"\n파일: {file_id}")
        print(f"  실제 라벨: {label_str} (idx={true_idx})")
        print(f"  예측 라벨: {pred_idx} ({idx_to_label.get(pred_idx, 'UNKNOWN')})")
        print(f"  확률 벡터: {probs}")
        print(f"  invoke 시간: {invoke_time*1000:.2f} ms")#입력 이미지 당 지연 시간
        print(f"  total_invoke 시간: {total_invoke_time*1000:.2f} ms")#전체 추론 시간
        print(f"  CPU 사용률 (invoke 구간): {cpu_percent:.1f}%")
        print(f"  램 사용 시작 용량 (system used): {invoke_start_used}")
        print(f"  램 사용 종료 용량 (system used): {invoke_end_used}")
        print(f"  램 사용 용량 (system diff): {ram_time}")
        print(f"  new 램 사용 시작 용량 (process RSS): {before}")
        print(f"  new 램 사용 종료 용량 (process RSS): {after}")
        print(f"  new 램 사용 용량 (process diff): {ram_diff}")

        # 7) 결과 저장 (나중에 DataFrame/CSV로 쓰기 좋게)
        results.append({
            "file_id": file_id,
            "true_label_str": label_str,
            "true_idx": true_idx,
            "pred_idx": pred_idx,
            "pred_label_str": idx_to_label.get(pred_idx, "UNKNOWN"),
            "probs": probs.tolist(),  # numpy → list로 저장
        })

        total += 1
        if pred_idx == true_idx:
            correct += 1

    # ★ 추가: 전체 추론 구간(전처리 + invoke 루프) 기준 측정 종료
    total_end_time = time.time()
    total_rss_after = proc_total.memory_info().rss
    total_ram_diff = total_rss_after - total_rss_before
    total_cpu_percent_all = proc_total.cpu_percent(interval=1,percpu=True)

    total_elapsed = total_end_time - total_start_time
    if total_elapsed > 0 and n_inference > 0:
        total_throughput = n_inference / total_elapsed          # samples/sec
        total_avg_latency_ms = (total_elapsed / n_inference) * 1000.0
    else:
        total_throughput = 0.0
        total_avg_latency_ms = 0.0

    print("\n==================== 전체 추론 구간 성능 지표 (루프 밖 측정) ====================")
    print(f"전체 추론 시간: {total_elapsed*1000:.3f} ms")
    print(f"전체 처리량: {total_throughput:.2f} samples/sec")
    print(f"입력 1개당 평균 지연시간: {total_avg_latency_ms:.3f} ms/sample")
    print(f"전체 추론 구간 프로세스 RSS 변화: {total_ram_diff / (1024**2):.2f} MB")
    print(f"전체 추론 구간 평균 CPU 사용률: {total_cpu_percent_all:.1f}%")
    print(f"  현재까지 최대 RSS: {peak_rss}")
    # 8) 전체 요약 (정답/총 개수)
    if total > 0:
        acc = correct / total
        print("\n==================== 요약 ====================")
        print(f"총 {total}개 중 {correct}개 정답 → 간단 정확도: {acc:.4f}")
    else:
        print("\n유효한 샘플이 없습니다.")

    # ★ 추가: 처리량(Throughput), 평균 invoke 시간, 평균 CPU 사용률
    if n_inference > 0 and total_invoke_time > 0:
        avg_invoke = total_invoke_time / n_inference
        throughput = n_inference / total_invoke_time
        avg_cpu = total_cpu_percent / n_inference

        print("\n==================== 성능 지표 (추론/처리량/CPU) ====================")
        print(f"평균 invoke 시간: {avg_invoke*1000:.3f} ms")
        print(f"처리량 (Throughput): {throughput:.2f} samples/sec")
        print(f"평균 CPU 사용률: {avg_cpu:.1f}%")

    # 9) 원하면 CSV로 저장 (선택)
    if len(results) > 0:
        try:
            import pandas as pd
            df_res = pd.DataFrame(results)
            df_res.to_csv("raspi_tflite_results.csv", index=False, encoding="utf-8-sig")
            print("\n결과가 'raspi_tflite_results.csv' 파일로 저장되었습니다.")
        except Exception as e:
            print(f"\nCSV 저장 중 오류 발생 (무시해도 됨): {e}")
        labels = [0, 1, 2]
        label_names = [idx_to_label[i] for i in labels]

        y_true = [r["true_idx"] for r in results]
        y_pred = [r["pred_idx"] for r in results]

        cm = confusion_matrix(y_true, y_pred, labels=labels)

        # multi-class라서 macro 평균 사용 (class별 지표 평균)
        precision = precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        recall    = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        f1        = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)

        # Specificity = TN / (TN + FP)  → class별 계산 후 macro 평균
        specificities = []
        for i, cls in enumerate(labels):
            TP = cm[i, i]
            FP = cm[:, i].sum() - TP
            FN = cm[i, :].sum() - TP
            TN = cm.sum() - (TP + FP + FN)
            spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0
            specificities.append(spec)

        specificity_macro = sum(specificities) / len(specificities)

        print("\n==================== 상세 지표 ====================")
        print("Confusion Matrix (행: 실제, 열: 예측):")
        print(cm)
        print("\n클래스 순서:", label_names)
        print(f"\nMacro Precision : {precision:.4f}")
        print(f"Macro Recall    : {recall:.4f}")
        print(f"\nMacro F1 Score  : {f1:.4f}")
        print(f"Macro Specificity: {specificity_macro:.4f}")

        # 원하면 class별 specificity도 같이 출력
        print("\n클래스별 Specificity:")
        for cls, spec in zip(label_names, specificities):
            print(f"  {cls}: {spec:.4f}")

    # 맨 마지막에 한 번 더 요약해서 보고 싶을 때
    print("\n마지막 invoke 기준 상세 로그:")
    print("시작 시간", invoke_start_time)
    print("종료 시간", invoke_end_time)
    print("걸린 시간", invoke_time)
    print("램 사용 시작 용량", invoke_start_used)
    print("램 사용 종료 용량", invoke_end_used)
    print("램 사용 용량", ram_time)
    print("new 램 사용 시작 용량", before)
    print("new 램 사용 종료 용량", after)
    print("new 램 사용 용량", ram_diff)

if __name__=="__main__":
    main()
