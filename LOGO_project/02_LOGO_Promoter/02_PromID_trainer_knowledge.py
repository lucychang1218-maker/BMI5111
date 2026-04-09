# -*- coding:utf-8 -*-
"""
Promoter training with "knowledge" (annotation channels / tracks)

✅ Supports FRACTION runs (frac5 / frac10) side-by-side
- Train/eval on *_frac5.npz and *_frac10.npz
- Output checkpoints/progress/metrics separated by frac (no overwrite)
- Resume per fold: skip completed folds if ckpt exists
- All outputs go to Google Drive paths (Colab-safe)
- Compute + eval metrics written into ONE metrics JSON per task per frac

✅ Extra safety:
- Auto-install biopython if missing (fix: ModuleNotFoundError: No module named 'Bio')
"""

import os
import sys
import warnings
import logging
import json
import time
import subprocess
from datetime import datetime

import numpy as np
from tqdm import tqdm

# =========================
# Global switches
# =========================
DEBUG = False

# =========================
# Colab / Drive paths  (⭐只需要改这里⭐)
# =========================
PROJECT_ROOT = "/content/drive/MyDrive/LOGO_project/test"
CHECKPOINT_ROOT = "/content/drive/MyDrive/LOGO_outputs/checkpoints"
METRICS_ROOT    = "/content/drive/MyDrive/LOGO_outputs/metrics"

# =========================
# Helper functions
# =========================
def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _safe_name(s: str) -> str:
    return s.replace("/", "_").replace("\\", "_").replace(" ", "_")

def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _progress_path(task_name: str, ngram: int, annotation_size: int, frac_tag: str) -> str:
    safe_task = _safe_name(task_name)
    return os.path.join(METRICS_ROOT, "progress", frac_tag,
                        f"progress_{safe_task}_{ngram}gram_{annotation_size}anno_{frac_tag}.json")

def _metrics_path(task_name: str, ngram: int, annotation_size: int, frac_tag: str) -> str:
    safe_task = _safe_name(task_name)
    return os.path.join(METRICS_ROOT, "metrics", frac_tag,
                        f"metrics_{safe_task}_{ngram}gram_{annotation_size}anno_{frac_tag}.json")

def _load_json_or_default(path: str, default: dict):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default

def _save_json_atomic(path: str, data: dict):
    _ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def _load_progress(progress_file: str):
    default = {
        "created_at": _now(),
        "updated_at": _now(),
        "completed_folds": [],
        "fold_results": {}
    }
    data = _load_json_or_default(progress_file, default)
    data.setdefault("created_at", _now())
    data.setdefault("updated_at", _now())
    data.setdefault("completed_folds", [])
    data.setdefault("fold_results", {})
    return data

def _save_progress(progress_file: str, data: dict):
    data["updated_at"] = _now()
    _save_json_atomic(progress_file, data)

def _to_float_list(eval_res):
    return [float(x) for x in eval_res]

def _maybe_install_biopython():
    """Fix: ModuleNotFoundError: No module named 'Bio' in Colab runtime."""
    try:
        import Bio  # noqa
        return
    except Exception:
        pass

    print("[Fix] biopython not found. Installing...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "biopython"])
        import Bio  # noqa
        print("[Fix] biopython installed OK.")
    except Exception as e:
        print("[Fix][ERROR] biopython install failed:", repr(e))
        print("You can run manually in a notebook cell:\n  !pip -q install biopython")
        raise

# =========================
# Environment / logs (MUST be before importing tensorflow)
# =========================
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
    absl_logging.set_stderrthreshold("error")
except Exception:
    pass

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Lambda
import tensorflow.keras.backend as K
from sklearn.model_selection import StratifiedKFold

# =========================
# Repo imports
# =========================
# 1) Ensure BioPython first (so repo import won't fail)
_maybe_install_biopython()

# 2) Put project root into sys.path so `from bgi...` works
sys.path.insert(0, PROJECT_ROOT)

from bgi.bert4keras.models import build_transformer_model
from bgi.common.refseq_utils import get_word_dict_for_n_gram_number

# =========================
# Annotation settings
# =========================
exclude_annotation = []  # e.g. [0, 3] to ablate channels

# =========================
# Compute monitor callback
# =========================
class ComputeMonitor(tf.keras.callbacks.Callback):
    """Record train timing stats."""
    def on_train_begin(self, logs=None):
        self._train_start = time.time()
        self._epoch_times = []

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self._epoch_times.append(time.time() - self._epoch_start)

    def on_train_end(self, logs=None):
        self.total_train_time_sec = float(time.time() - self._train_start)
        self.epochs_ran = int(len(self._epoch_times))
        self.avg_epoch_time_sec = float(np.mean(self._epoch_times)) if self._epoch_times else 0.0

def _get_gpu_peak_memory_gb():
    """Best-effort GPU peak memory. Return None if not supported / no GPU."""
    try:
        if not tf.config.list_physical_devices("GPU"):
            return None
        info = tf.config.experimental.get_memory_info("GPU:0")
        peak_bytes = info.get("peak", None)
        if peak_bytes is None:
            return None
        return float(peak_bytes) / (1024 ** 3)
    except Exception:
        return None

# =========================
# Metrics
# =========================
def f1_score(y_true, y_pred):
    y_true = tf.cast(y_true, "float32")
    y_pred = tf.cast(y_pred, "float32")
    tp = K.sum(tf.cast(K.equal(y_true, 1) & K.equal(K.round(y_pred), 1), "float32"))
    fp = K.sum(tf.cast(K.equal(y_true, 0) & K.equal(K.round(y_pred), 1), "float32"))
    fn = K.sum(tf.cast(K.equal(y_true, 1) & K.equal(K.round(y_pred), 0), "float32"))
    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())
    return 2 * p * r / (p + r + K.epsilon())

# =========================
# Data Loading
# =========================
def load_npz_data_for_classification(file_name, ngram=5, only_one_slice=True, ngram_index=None):
    """
    Expect npz contains:
      - sequence: (N, L)
      - annotation: (N, C, L)
      - label: (N,)
    """
    if (not str(file_name).endswith(".npz")) or (not os.path.exists(file_name)):
        raise FileNotFoundError(f"NPZ not found: {os.path.abspath(file_name)}")

    loaded = np.load(file_name, allow_pickle=True)
    x_data = loaded["sequence"]
    anno_data = loaded["annotation"]
    y_data = loaded["label"]

    if DEBUG:
        print("Load:", file_name)
        print("X:", x_data.shape, "Anno:", anno_data.shape, "Y:", y_data.shape)

    x_all, anno_all, y_all = [], [], []
    if only_one_slice:
        for ii in range(ngram):
            if ngram_index is not None and ii != ngram_index:
                continue
            kk = ii
            slice_indexes = []
            max_slice_seq_len = x_data.shape[1] // ngram * ngram
            for gg in range(kk, max_slice_seq_len, ngram):
                slice_indexes.append(gg)

            x_slice = x_data[:, slice_indexes]
            anno_slice = anno_data[:, :, slice_indexes]  # (N, C, L/ngram)

            x_all.append(x_slice)
            anno_all.append(anno_slice)
            y_all.append(y_data)
    else:
        x_all.append(x_data)
        anno_all.append(anno_data)
        y_all.append(y_data)

    return x_all, anno_all, y_all

def load_all_data(record_names, ngram=5, only_one_slice=True, ngram_index=None):
    x_all, anno_all, y_all = [], [], []
    for fn in record_names:
        x_list, a_list, y_list = load_npz_data_for_classification(
            fn, ngram=ngram, only_one_slice=only_one_slice, ngram_index=ngram_index
        )
        x_all.extend(x_list)
        anno_all.extend(a_list)
        y_all.extend(y_list)

    X = np.concatenate(x_all, axis=0)
    A = np.concatenate(anno_all, axis=0)
    Y = np.concatenate(y_all, axis=0)
    return X, A, Y

def make_tf_dataset(X_seq, A_anno, Y, promoter_seq_len, annotation_size, shuffle=False):
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    def gen():
        n = len(X_seq)
        idxs = np.arange(n)
        if shuffle:
            np.random.shuffle(idxs)
        i = 0
        while True:
            if i >= n:
                i = 0
                idxs = np.arange(n)
                if shuffle:
                    np.random.shuffle(idxs)

            idx = idxs[i]
            i += 1

            x = X_seq[idx].astype(np.int32)
            seg = np.zeros_like(x, dtype=np.int32)
            anno = A_anno[idx].astype(np.int32)  # (C, L')
            y = Y[idx].astype(np.float32)
            yield x, seg, anno, y

    ds = tf.data.Dataset.from_generator(
        gen,
        output_types=(tf.int32, tf.int32, tf.int32, tf.float32),
        output_shapes=(
            tf.TensorShape([promoter_seq_len]),
            tf.TensorShape([promoter_seq_len]),
            tf.TensorShape([annotation_size, promoter_seq_len]),
            tf.TensorShape([1]),
        ),
    )
    return ds

def parse_function(x_token, x_seg, annotations, y, annotation_size):
    inputs = {
        "Input-Token": tf.cast(x_token, tf.int32),
        "Input-Segment": tf.cast(x_seg, tf.int32),
    }

    for ii in range(annotation_size):
        gene_type = annotations[:, ii, :]  # (B, L)
        if ii in exclude_annotation:
            gene_type = tf.zeros_like(gene_type, dtype=tf.int32)
        inputs[f"Input-Token-Type_{ii}"] = tf.cast(gene_type, tf.int32)

    y = tf.cast(y, tf.float32)
    return inputs, y

# =========================
# Model
# =========================
def model_def_with_annotation_size(
    vocab_size,
    annotation_size,
    embedding_size=128,
    hidden_size=256,
    num_heads=8,
    num_hidden_layers=2,
    intermediate_size=1024,
    max_position_embeddings=512,
    drop_rate=0.25,
):
    multi_inputs = [2] * annotation_size

    config = {
        "attention_probs_dropout_prob": 0,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0,
        "embedding_size": embedding_size,
        "hidden_size": hidden_size,
        "initializer_range": 0.02,
        "intermediate_size": intermediate_size,
        "max_position_embeddings": max_position_embeddings,
        "num_attention_heads": num_heads,
        "num_hidden_layers": num_hidden_layers,
        "num_hidden_groups": 1,
        "net_structure_type": 0,
        "gap_size": 0,
        "num_memory_blocks": 0,
        "inner_group_num": 1,
        "down_scale_factor": 1,
        "type_vocab_size": 0,
        "vocab_size": vocab_size,
        "custom_masked_sequence": False,
        "custom_conv_layer": True,
        "use_segment_ids": True,
        "use_position_ids": True,
        "multi_inputs": multi_inputs,
    }

    bert = build_transformer_model(
        configs=config,
        model="multi_inputs_bert",
        return_keras_model=False,
    )

    cls = Lambda(lambda x: x[:, 0])(bert.model.output)
    out = BatchNormalization()(cls)
    out = Dropout(drop_rate)(out)
    out = Dense(1, activation="sigmoid", name="CLS-Activation")(out)
    return tf.keras.models.Model(inputs=bert.model.input, outputs=out)

# =========================
# Training
# =========================
def train_kfold(
    train_data_file,
    data_path,
    annotation_size,
    vocab_size,
    frac_tag,
    pretrained_weight=None,
    batch_size=128,
    epochs=50,
    ngram=5,
    n_splits=10,
    PROMOTER_RESIZED_LEN=600,
    task_name="epdnew_Knowledge",
    early_patience=10,
    min_lr=1e-6,
):
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) >= 2:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.get_strategy()

    GLOBAL_BATCH_SIZE = batch_size * max(1, strategy.num_replicas_in_sync)
    num_parallel_calls = 16
    only_one_slice = True

    train_files = [os.path.join(data_path, train_data_file)]
    if not os.path.exists(train_files[0]):
        raise FileNotFoundError(f"[{task_name}/{frac_tag}] file not found: {os.path.abspath(train_files[0])}")

    # load data
    X, A, Y = load_all_data(train_files, ngram=ngram, only_one_slice=only_one_slice, ngram_index=None)

    # seeds (fixed)
    seed = 7
    np.random.seed(seed)
    tf.random.set_seed(seed)

    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    promoter_seq_len = (PROMOTER_RESIZED_LEN // ngram * ngram)
    if only_one_slice:
        promoter_seq_len = promoter_seq_len // ngram

    _ensure_dir(CHECKPOINT_ROOT)
    _ensure_dir(METRICS_ROOT)

    task_out_dir = os.path.join(
        CHECKPOINT_ROOT, frac_tag, f"{_safe_name(task_name)}_{ngram}gram_{annotation_size}anno_{frac_tag}"
    )
    _ensure_dir(task_out_dir)

    progress_file = _progress_path(task_name, ngram, annotation_size, frac_tag)
    progress = _load_progress(progress_file)
    completed = set(progress.get("completed_folds", []))
    fold_results = progress.get("fold_results", {})

    metric_file = _metrics_path(task_name, ngram, annotation_size, frac_tag)
    metrics_all = _load_json_or_default(metric_file, {
        "created_at": _now(),
        "updated_at": _now(),
        "task_name": task_name,
        "frac_tag": frac_tag,
        "ngram": ngram,
        "annotation_size": annotation_size,
        "train_data_file": train_data_file,
        "data_path": os.path.abspath(data_path),
        "pretrained_weight": os.path.abspath(pretrained_weight) if pretrained_weight else None,
        "folds": {}
    })

    print("\n" + "=" * 90)
    print(f"[RUN] task={task_name}  frac={frac_tag}  ngram={ngram}  anno={annotation_size}")
    print(f"[NPZ] {os.path.abspath(train_files[0])}")
    print(f"[Checkpoint Dir] {os.path.abspath(task_out_dir)}")
    print(f"[Progress File]  {os.path.abspath(progress_file)}")
    print(f"[Metric File]    {os.path.abspath(metric_file)}")
    print(f"[Resume] completed folds = {sorted(completed)}")
    print("=" * 90)

    for fold, (train_idx, test_idx) in tqdm(
        enumerate(kfold.split(X, Y)),
        total=n_splits,
        desc=f"KFold({task_name},{frac_tag})",
        dynamic_ncols=True,
        leave=True,
    ):
        ckpt_path = os.path.join(
            task_out_dir,
            f"promoter_best_model_{_safe_name(task_name)}_{frac_tag}_fold{fold}_{ngram}gram_{annotation_size}anno.h5"
        )

        # ✅ resume
        if fold in completed and os.path.exists(ckpt_path):
            continue

        tf.keras.backend.clear_session()

        # split train -> train/valid (90/10)
        promoter_indexes = np.arange(len(train_idx))
        np.random.shuffle(promoter_indexes)
        tr_slice = promoter_indexes[: int(len(promoter_indexes) * 0.9)]
        va_slice = promoter_indexes[int(len(promoter_indexes) * 0.9):]

        x_train = X[train_idx[tr_slice]]
        a_train = A[train_idx[tr_slice]]
        y_train = Y[train_idx[tr_slice]]

        x_valid = X[train_idx[va_slice]]
        a_valid = A[train_idx[va_slice]]
        y_valid = Y[train_idx[va_slice]]

        x_test = X[test_idx]
        a_test = A[test_idx]
        y_test = Y[test_idx]

        train_steps = max(1, len(y_train) // GLOBAL_BATCH_SIZE)
        valid_steps = max(1, len(y_valid) // GLOBAL_BATCH_SIZE)
        test_steps  = max(1, len(y_test)  // GLOBAL_BATCH_SIZE)

        compute_cb = ComputeMonitor()

        callbacks = [
            compute_cb,
            ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True, verbose=0),
            EarlyStopping(monitor="val_loss", patience=early_patience, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=max(1, early_patience // 3),
                min_lr=min_lr,
                verbose=0,
            ),
        ]

        train_ds = make_tf_dataset(x_train, a_train, y_train, promoter_seq_len, annotation_size, shuffle=True)
        train_ds = train_ds.shuffle(len(y_train), reshuffle_each_iteration=True)
        train_ds = train_ds.batch(batch_size)
        train_ds = train_ds.map(lambda x, s, a, y: parse_function(x, s, a, y, annotation_size),
                                num_parallel_calls=num_parallel_calls)
        train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

        valid_ds = make_tf_dataset(x_valid, a_valid, y_valid, promoter_seq_len, annotation_size, shuffle=False)
        valid_ds = valid_ds.batch(batch_size)
        valid_ds = valid_ds.map(lambda x, s, a, y: parse_function(x, s, a, y, annotation_size),
                                num_parallel_calls=num_parallel_calls)
        valid_ds = valid_ds.prefetch(tf.data.experimental.AUTOTUNE)

        test_ds = make_tf_dataset(x_test, a_test, y_test, promoter_seq_len, annotation_size, shuffle=False)
        test_ds = test_ds.batch(batch_size)
        test_ds = test_ds.map(lambda x, s, a, y: parse_function(x, s, a, y, annotation_size),
                              num_parallel_calls=num_parallel_calls)
        test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)

        with strategy.scope():
            model = model_def_with_annotation_size(
                vocab_size=vocab_size,
                annotation_size=annotation_size,
                embedding_size=128,
                hidden_size=256,
                num_heads=8,
                num_hidden_layers=2,
                intermediate_size=1024,
                max_position_embeddings=512,
                drop_rate=0.25,
            )
            model.compile(
                loss="binary_crossentropy",
                optimizer=Adam(learning_rate=1e-4),
                metrics=[
                    "acc",
                    tf.keras.metrics.Precision(name="precision"),
                    tf.keras.metrics.Recall(name="recall"),
                    f1_score,
                ],
            )

            if pretrained_weight and os.path.exists(pretrained_weight):
                model.load_weights(pretrained_weight, by_name=True, skip_mismatch=True)

            total_params = int(model.count_params())
            trainable_params = int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))
            trainable_ratio = float(trainable_params) / float(total_params) if total_params > 0 else 0.0

        model.fit(
            train_ds,
            steps_per_epoch=train_steps,
            epochs=epochs,
            validation_data=valid_ds,
            validation_steps=valid_steps,
            callbacks=callbacks,
            verbose=0,
        )

        # eval with best ckpt
        with strategy.scope():
            model2 = model_def_with_annotation_size(
                vocab_size=vocab_size,
                annotation_size=annotation_size,
                embedding_size=128,
                hidden_size=256,
                num_heads=8,
                num_hidden_layers=2,
                intermediate_size=1024,
                max_position_embeddings=512,
                drop_rate=0.25,
            )
            model2.compile(
                loss="binary_crossentropy",
                optimizer=Adam(learning_rate=1e-4),
                metrics=[
                    "acc",
                    tf.keras.metrics.Precision(name="precision"),
                    tf.keras.metrics.Recall(name="recall"),
                    f1_score,
                ],
            )
            model2.load_weights(ckpt_path)

        eval_res = model2.evaluate(test_ds, steps=test_steps, verbose=0)
        eval_list = _to_float_list(eval_res)
        loss_v, acc_v, prec_v, rec_v, f1_v = eval_list[0], eval_list[1], eval_list[2], eval_list[3], eval_list[4]

        print(f"[Eval] {task_name} {frac_tag} Fold {fold}: acc={acc_v:.4f} precision={prec_v:.4f} recall={rec_v:.4f} f1={f1_v:.4f}")

        gpu_peak_gb = _get_gpu_peak_memory_gb()

        # progress json
        fold_key = f"fold_{fold}"
        fold_results[fold_key] = {
            "time": _now(),
            "ckpt_path": os.path.abspath(ckpt_path),
            "metrics": {
                "loss": loss_v,
                "acc": acc_v,
                "precision": prec_v,
                "recall": rec_v,
                "f1": f1_v,
                "raw_list": eval_list
            }
        }

        completed.add(fold)
        progress["completed_folds"] = sorted(completed)
        progress["fold_results"] = fold_results
        _save_progress(progress_file, progress)

        # metrics json
        metrics_all.setdefault("folds", {})
        metrics_all["updated_at"] = _now()
        metrics_all["folds"][fold_key] = {
            "time": _now(),
            "paths": {
                "ckpt_path": os.path.abspath(ckpt_path),
                "progress_json": os.path.abspath(progress_file),
            },
            "data_summary": {
                "train": int(len(y_train)),
                "valid": int(len(y_valid)),
                "test": int(len(y_test)),
                "steps": {
                    "train_steps": int(train_steps),
                    "valid_steps": int(valid_steps),
                    "test_steps": int(test_steps),
                },
                "batch_size": int(batch_size),
                "global_batch_size": int(GLOBAL_BATCH_SIZE),
            },
            "model_complexity": {
                "total_params": total_params,
                "trainable_params": trainable_params,
                "trainable_ratio": float(trainable_ratio),
            },
            "training_resource": {
                "epochs_ran": int(getattr(compute_cb, "epochs_ran", 0)),
                "avg_epoch_time_sec": float(getattr(compute_cb, "avg_epoch_time_sec", 0.0)),
                "total_train_time_sec": float(getattr(compute_cb, "total_train_time_sec", 0.0)),
            },
            "gpu": {
                "peak_memory_gb": gpu_peak_gb,
            },
            "eval_metrics": {
                "loss": float(loss_v),
                "acc": float(acc_v),
                "precision": float(prec_v),
                "recall": float(rec_v),
                "f1": float(f1_v),
                "raw_list": eval_list,
            },
        }
        _save_json_atomic(metric_file, metrics_all)
        print(f"[Fold Done] {task_name} {frac_tag} fold {fold} -> wrote progress + metrics json.")

# =========================
# Main
# =========================
if __name__ == "__main__":
    # GPU memory growth
    gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    # cd to promoter folder so ./data/... works
    promoter_dir = os.path.join(PROJECT_ROOT, "02_LOGO_Promoter")
    os.chdir(promoter_dir)
    print("[CWD]", os.getcwd())

    ngram = 5
    word_dict = get_word_dict_for_n_gram_number(n_gram=ngram)
    vocab_size = len(word_dict) + 10

    pretrained_weight = None
    # pretrained_weight = "/content/drive/MyDrive/LOGO_project/test/99_PreTrain_Model_Weight/xxx.hdf5"

    FRAC_TAGS = ["frac5", "frac10"]

    # ✅ 三任务
    TASKS = [
        {"base_task_name": "epdnew_BOTH_Knowledge",        "anno_size": 13, "batch": 128},
        {"base_task_name": "epdnew_NO_TATA_BOX_Knowledge", "anno_size": 13, "batch": 256},
        {"base_task_name": "epdnew_TATA_BOX_Knowledge",    "anno_size": 13, "batch": 128},
    ]

    # ✅ 你要求：跳过已经跑好的 TATA+
    # 如果你想只跳过某个 frac（比如只跳过 TATA+ frac5），也能按下面方式改。
    SKIP_TASKS = {"epdnew_TATA_BOX_Knowledge"}  # <- 你现在就用这个

    data_path = f"./data/{ngram}_gram_11_knowledge"

    for frac_tag in FRAC_TAGS:
        for t in tqdm(TASKS, desc=f"Tasks({frac_tag})", total=len(TASKS), dynamic_ncols=True):
            base = t["base_task_name"]

            if base in SKIP_TASKS:
                print(f"[SKIP] {base} (already done) for {frac_tag}")
                continue

            train_file = f"{base}_{ngram}_gram_{frac_tag}.npz"
            train_kfold(
                train_data_file=train_file,
                data_path=data_path,
                annotation_size=t["anno_size"],
                vocab_size=vocab_size,
                frac_tag=frac_tag,
                pretrained_weight=pretrained_weight,
                batch_size=t["batch"],
                epochs=50,
                early_patience=10,
                ngram=ngram,
                n_splits=10,
                task_name=base,
            )
