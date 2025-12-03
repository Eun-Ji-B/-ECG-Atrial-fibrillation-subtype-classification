
# ★ 추가: 현재 프로세스 객체
proc = psutil.Process(os.getpid())

# =========================================================
# 0. 기본 설정 (df, train_df, val_df, test_df, H, W 는 이미 준비되어 있다고 가정)
# =========================================================
CLASSES = 3  # 0: Non, 1: PAF, 2: PsAF
EPOCHS = 60
BATCH_SIZE = 256

# -------------------------------------------------
# 1. DataFrame 그대로 사용해서 NumPy 배열 만들기 (tqdm 적용)
# -------------------------------------------------
def build_Xy_from_df(split_df, H, W, desc=""):
    x_list, y_list = [], []

    # tqdm으로 진행 상황 표시
    for row in tqdm(split_df.itertuples(index=False),
                    total=len(split_df),
                    desc=desc):
        arr = np.load(row.path).astype(np.float32)   # (H, W) 또는 (H, W, 1) 가정

        # 2D이면 채널 축 추가
        if arr.ndim == 2:
            arr = arr[..., np.newaxis]              # (H, W, 1)

        x_list.append(arr)
        y_list.append(int(row.label))

    X = np.stack(x_list, axis=0)               # (N, H, W, 1)
    y = np.array(y_list, dtype=np.int32)       # (N,)
    return X, y



print("\n--- NumPy 배열 생성 (train/val/test) ---")
X_train, y_train = build_Xy_from_df(train_pids_new, H, W, desc="Build Train")
X_val,   y_val   = build_Xy_from_df(val_pids_new,   H, W, desc="Build Val")
X_test,  y_test  = build_Xy_from_df(test_pids_new,  H, W, desc="Build Test")

print("X_train shape:", X_train.shape, "y_train shape:", y_train.shape)
print("X_val   shape:", X_val.shape,   "y_val shape:",   y_val.shape)
print("X_test  shape:", X_test.shape,  "y_test shape:",  y_test.shape)

# ==============================
# ★ 배열을 파일로 저장 (한 번만 실행하면 됨)
# ==============================
import os
import numpy as np

SAVE_DIR = "/content/drive/MyDrive/2학기프로젝트/jeuni/Re심전도 데이터/1202AF_project_model_data/cache_arrays"  # 원하는 경로로 바꾸세요
os.makedirs(SAVE_DIR, exist_ok=True)

save_path = os.path.join(SAVE_DIR, "af_arrays_cqt_qdeep.npz")

np.savez_compressed(
    save_path,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    X_test=X_test,
    y_test=y_test,
)

print(f"\n배열을 저장했습니다: {save_path}")


