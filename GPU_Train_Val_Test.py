# ====================== 여기까지는 네가 준 코드 그대로 ======================
# df, train_df, val_df, test_df, H, W, train_ds/val_ds/test_ds 까지 생성된 상태

from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Add, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tqdm.auto import tqdm   # 진행 상황 표시용
import time

import os
import numpy as np
import psutil   # ★ 추가: CPU/RAM 지표 측정용


import numpy as np
import os

SAVE_DIR = "/content/drive/MyDrive/2학기프로젝트/jeuni/Re심전도 데이터/1202AF_project_model_data/cache_arrays"  # 위에서 사용한 것과 동일해야 함
save_path = os.path.join(SAVE_DIR, "af_arrays_cqt_qdeep.npz")

data = np.load(save_path)

X_train = data["X_train"]
y_train = data["y_train"]
X_val   = data["X_val"]
y_val   = data["y_val"]
X_test  = data["X_test"]
y_test  = data["y_test"]

print("로드 완료!")
print("X_train shape:", X_train.shape, "y_train shape:", y_train.shape)
print("X_val   shape:", X_val.shape,   "y_val shape:",   y_val.shape)
print("X_test  shape:", X_test.shape,  "y_test shape:",  y_test.shape)

# =========================================================
# 1. 기본 설정 (df, train_df, val_df, test_df, H, W 는 이미 준비되어 있다고 가정)
# =========================================================
CLASSES = 3  # 0: Non, 1: PAF, 2: PsAF
EPOCHS = 60
BATCH_SIZE = 256
H = 80
W = 1001
proc = psutil.Process(os.getpid())
# -------------------------------------------------
# 2. Q-Deep 블록 정의 (identity / convolutional)
# -------------------------------------------------
def identity_block(X, f, filters):
    F1, F2 = filters
    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(f, f), strides=(1, 1), padding='same')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same')(X)
    X = BatchNormalization(axis=3)(X)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    return X

def convolutional_block(X, f, filters, s=2):
    F1, F2 = filters
    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(f, f), strides=(s, s), padding='same')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same')(X)
    X = BatchNormalization(axis=3)(X)

    X_shortcut = Conv2D(filters=F2, kernel_size=(1, 1), strides=(s, s), padding='same')(X_shortcut)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    return X

def QDeepModel(input_shape, classes=3):
    X_input = Input(input_shape)

    # Conv1 (원 논문 padding=6 대신 Keras에서 'same' 사용)
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', padding='same')(X_input)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(X)

    # CNN Block: Layer 0 ~ 5
    # Layer 0 (Identity only): 64 filters
    X = identity_block(X, f=3, filters=[64, 64])

    # Layer 1: 64 filters (유지)
    # 첫 번째 Conv Block은 보통 사이즈를 줄이지 않기도 함 (s=1)
    X = convolutional_block(X, f=3, filters=[64, 64], s=1)
    X = identity_block(X, f=3, filters=[64, 64])

    # Layer 2: 128 filters (확장 & 다운샘플링)
    X = convolutional_block(X, f=3, filters=[128, 128], s=2) # s=2로 이미지 크기 축소
    X = identity_block(X, f=3, filters=[128, 128])

    # Layer 3: 256 filters
    X = convolutional_block(X, f=3, filters=[256, 256], s=2)
    X = identity_block(X, f=3, filters=[256, 256])

    # Layer 4: 512 filters
    X = convolutional_block(X, f=3, filters=[512, 512], s=2)
    X = identity_block(X, f=3, filters=[512, 512])

    # Layer 5: 512 filters (유지) - 논문의 32채널 오류 수정
    X = convolutional_block(X, f=3, filters=[512, 512], s=2)
    X = identity_block(X, f=3, filters=[512, 512])

    # GAP & Output
    X = GlobalAveragePooling2D(name='gap')(X)
    X = Dense(512, activation='relu', name='fc_512')(X)
    X = Dense(4,   activation='relu', name='fc_4')(X)
    X = Dense(classes, activation='softmax', name='fc_out')(X)

    model = Model(inputs=X_input, outputs=X, name='QDeepModel_Optimized')
    return model

# -------------------------------------------------
# 3. tqdm으로 에폭 진행 상황을 보여주는 콜백
# -------------------------------------------------
class TqdmEpochCallback(Callback):
    def on_train_begin(self, logs=None):
        # 전체 에폭 수만큼 진행 바 생성
        self.epochs_bar = tqdm(total=self.params['epochs'],
                               desc="Training epochs")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        postfix = {
            "loss":     f"{logs.get('loss', 0):.4f}",
            "val_loss": f"{logs.get('val_loss', 0):.4f}",
            "acc":      f"{logs.get('accuracy', 0):.4f}",
            "val_acc":  f"{logs.get('val_accuracy', 0):.4f}",
        }
        self.epochs_bar.set_postfix(postfix)
        self.epochs_bar.update(1)

    def on_train_end(self, logs=None):
        self.epochs_bar.close()


# -------------------------------------------------
# 4. 모델 생성 및 컴파일
# -------------------------------------------------
INPUT_SHAPE = (H, W, 1)

MODEL_DIR = "/content/drive/MyDrive/2학기프로젝트/jeuni/Re심전도 데이터/REREQDeep_models1202"
os.makedirs(MODEL_DIR, exist_ok=True)

print("\n--- Q-Deep 모델 생성 ---")
model = QDeepModel(input_shape=INPUT_SHAPE, classes=CLASSES)
model.summary()
#model.load_state_dict(torch.load('/content/drive/MyDrive/2학기프로젝트/jeuni/Re심전도 데이터/REREQDeep_models1129/model_weights.pth'))

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',   # y가 정수 레이블(0,1,2)이니까 sparse 사용
    metrics=['accuracy']
)


# -------------------------------------------------
# 5. 모델 저장 경로 & 콜백 설정
# -------------------------------------------------

best_model_path = os.path.join(MODEL_DIR, "QDeep_best_val_loss.keras")
last_model_path = os.path.join(MODEL_DIR, "QDeep_last_epoch.keras")  # 마지막 epoch 가중치용 (옵션)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    filepath=best_model_path,
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

checkpoint_last = ModelCheckpoint(
    filepath=last_model_path,
    monitor='val_loss',
    save_best_only=False,
    save_weights_only=False,
    verbose=1
)


# -------------------------------------------------
# 6. 학습 (tqdm 에폭 진행 바 사용)
# -------------------------------------------------
print("\n--- 모델 학습 시작 ---")

# ★ 추가: train 구간 CPU/RSS 측정 준비
_ = proc.cpu_percent(interval=None)
train_rss_before = proc.memory_info().rss

train_start_time = time.time()
trian_start_used = check_ram_usage()
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    callbacks=[early_stop, checkpoint, checkpoint_last, TqdmEpochCallback()],
    verbose=0   # Keras 기본 로그 끄고 tqdm만 사용
)
train_end_time = time.time()
trian_end_used = check_ram_usage()

# ★ 추가: train 구간 CPU/RSS 결과
train_cpu_percent = proc.cpu_percent(interval=None)
train_rss_after = proc.memory_info().rss
train_rss_diff = train_rss_after - train_rss_before

# -------------------------------------------------
# 7. 테스트 평가
# -------------------------------------------------
print("\n--- 테스트 데이터 평가 ---")

# ★ 추가: eval 구간 CPU/RSS 측정 준비
_ = proc.cpu_percent(interval=None)
eval_rss_before = proc.memory_info().rss

eval_start_time = time.time()
eval_start_used = check_ram_usage()
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
eval_end_time = time.time()
eval_end_used = check_ram_usage()

# ★ 추가: eval 구간 CPU/RSS 결과
eval_cpu_percent = proc.cpu_percent(interval=None)
eval_rss_after = proc.memory_info().rss
eval_rss_diff = eval_rss_after - eval_rss_before

# ★ 추가: predict 구간 CPU/RSS 측정 준비
_ = proc.cpu_percent(interval=None)
pred_rss_before = proc.memory_info().rss

predict_start_time = time.time()
predict_start_used = check_ram_usage()
y_prob = model.predict(X_test, verbose=0)
predict_end_time = time.time()
predict_end_used = check_ram_usage()

# ★ 추가: predict 구간 CPU/RSS 결과
predict_cpu_percent = proc.cpu_percent(interval=None)
pred_rss_after = proc.memory_info().rss
pred_rss_diff = pred_rss_after - pred_rss_before

# ★ 추가: GPU 모델 추론 처리량/평균 지연시간 (X_test 전체 기준)
predict_time = predict_end_time - predict_start_time
if predict_time > 0 and len(X_test) > 0:
    gpu_throughput = len(X_test) / predict_time          # samples/sec
    gpu_avg_latency_ms = (predict_time / len(X_test)) * 1000.0
else:
    gpu_throughput = 0.0
    gpu_avg_latency_ms = 0.0

y_pred = np.argmax(y_prob, axis=1)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

cm = confusion_matrix(y_test, y_pred)
specificity_list = []
for i in range(CLASSES):
    fp = cm[:, i].sum() - cm[i, i]
    tn = cm.sum() - (cm[:, i].sum() + cm[i, :].sum() - cm[i, i])
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    specificity_list.append(spec)
specificity = np.mean(specificity_list)

print("\n===== 최종 테스트 결과 =====")
print(f"Test Loss      : {test_loss:.4f}")
print(f"Accuracy       : {acc:.4f}")
print(f"Precision      : {prec:.4f}")
print(f"Recall         : {rec:.4f}")
print(f"Specificity    : {specificity:.4f}")
print(f"F1 Score       : {f1:.4f}")
print("Confusion Matrix:\n", cm)

print("train 걸린 시간 : ", train_end_time - train_start_time)
print("eval 걸린 시간 :", eval_end_time - eval_start_time)
print("predict 걸린 시간 : ", predict_end_time - predict_start_time)

print(f"train 사용 램 용량 : {(trian_end_used - trian_start_used)/(1024**3):.2f}")
print(f"eval 사용 램 용량 :{(eval_end_used - eval_start_used)/(1024**3):.2f}")
print(f"predict 사용 램 용량 : {(predict_end_used - predict_start_used)/(1024**3):.2f}")

# ★ 추가: CPU / 프로세스 RSS / 처리량 지표 요약
print("\n===== 추가 리소스/성능 지표 (GPU Keras 모델) =====")
print(f"train 평균 CPU 사용률 : {train_cpu_percent:.1f}%")
print(f"eval  평균 CPU 사용률 : {eval_cpu_percent:.1f}%")
print(f"predict 평균 CPU 사용률 : {predict_cpu_percent:.1f}%")

print(f"train 프로세스 RSS 변화 : {train_rss_diff / (1024**2):.2f} MB")
print(f"eval  프로세스 RSS 변화 : {eval_rss_diff / (1024**2):.2f} MB")
print(f"predict 프로세스 RSS 변화 : {pred_rss_diff / (1024**2):.2f} MB")

print(f"GPU 모델 추론 처리량 (X_test 기준) : {gpu_throughput:.2f} samples/sec")
print(f"GPU 모델 평균 추론 지연시간 : {gpu_avg_latency_ms:.3f} ms/sample")

# -------------------------------------------------
# 8. 모델 저장
# -------------------------------------------------
final_model_path = os.path.join(MODEL_DIR, "QDeep_final_after_eval.keras")
model.save(final_model_path)
print(f"\n최종 모델이 다음 경로에 저장되었습니다:\n{final_model_path}")

# ★ 추가: .keras 파일 크기 (GPU 모델 크기 지표)
if os.path.exists(final_model_path):
    final_size_bytes = os.path.getsize(final_model_path)
    final_size_mb = final_size_bytes / (1024**2)
    print(f"[INFO] 최종 Keras 모델 파일 크기: {final_size_mb:.2f} MB")

# Keras 3 / TF 버전에 따라 model.export 가 없으면 이 부분은 주석 처리하세요.
try:
    model.export('/content/drive/MyDrive/2학기프로젝트/jeuni/Re심전도 데이터/REREQDeep_models1129/My_Model')
    print("\nmodel.export로 내보내기 완료")
except AttributeError:
    print("\n현재 Keras/TF 버전에서는 model.export 를 사용할 수 없습니다. (무시해도 됩니다.)")
