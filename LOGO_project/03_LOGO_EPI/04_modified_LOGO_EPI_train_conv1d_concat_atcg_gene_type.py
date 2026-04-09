# -*- coding:utf-8 -*-
"""
04_clean_LOGO_EPI_train_conv1d_concat_atcg_frac_oldcompare13.py

Old-style compare training:
- accepts --frac like your previous usage
- force annotation_size=13
- keep old floor-truncated validation/test steps
- add full-test AUC/AUPRC for comparison
"""
import os
import sys
import gc
import json
import warnings
import logging
from datetime import datetime

import numpy as np
from tqdm import tqdm

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import tensorflow.keras.backend as K
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

PROJECT_ROOT = "/content/drive/MyDrive/LOGO_project/test"
DATA_ROOT = "/content/drive/MyDrive/LOGO_project/test/03_LOGO_EPI/data/data"
CHECKPOINT_ROOT = "/content/drive/MyDrive/LOGO_outputs/checkpoints_epi_oldcompare13_frac"
METRICS_ROOT = "/content/drive/MyDrive/LOGO_outputs/metrics_epi_oldcompare13_frac"

sys.path.insert(0, PROJECT_ROOT)
from bgi.bert4keras.models import build_transformer_model
from bgi.common.refseq_utils import get_word_dict_for_n_gram_number


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _safe_name(s: str) -> str:
    return s.replace("/", "_").replace("\\", "_").replace(" ", "_")


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


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


def f1_score(y_true, y_pred):
    y_true = tf.cast(y_true, "float32")
    y_pred = tf.cast(y_pred, "float32")
    tp = K.sum(tf.cast(K.equal(y_true, 1) & K.equal(K.round(y_pred), 1), "float32"))
    fp = K.sum(tf.cast(K.equal(y_true, 0) & K.equal(K.round(y_pred), 1), "float32"))
    fn = K.sum(tf.cast(K.equal(y_true, 1) & K.equal(K.round(y_pred), 0), "float32"))
    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())
    return 2 * p * r / (p + r + K.epsilon())


def load_all_data(record_names):
    x_all, a_all, y_all = [], [], []
    for fn in record_names:
        loaded = np.load(fn, allow_pickle=True)
        x_all.append(np.asarray(loaded["sequence"]))
        a_all.append(np.asarray(loaded["annotation"]))
        y_all.append(np.asarray(loaded["label"]))
    X = np.concatenate(x_all, axis=0)
    A = np.concatenate(a_all, axis=0)
    Y = np.concatenate(y_all, axis=0)
    return X, A, Y


def make_tf_dataset(X_enh, A_enh, X_pro, A_pro, Y, enhancer_seq_len, promoter_seq_len, annotation_size, shuffle=False):
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    def gen():
        n = len(Y)
        idxs = np.arange(n)
        while True:
            if shuffle:
                np.random.shuffle(idxs)
            for idx in idxs:
                x_enh = X_enh[idx].astype(np.int32)
                a_enh = A_enh[idx].astype(np.int32)
                x_pro = X_pro[idx].astype(np.int32)
                a_pro = A_pro[idx].astype(np.int32)
                seg_enh = np.zeros_like(x_enh, dtype=np.int32)
                seg_pro = np.zeros_like(x_pro, dtype=np.int32)
                y = Y[idx].astype(np.float32)
                yield x_enh, a_enh, seg_enh, x_pro, a_pro, seg_pro, y

    ds = tf.data.Dataset.from_generator(
        gen,
        output_types=(tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.float32),
        output_shapes=(
            tf.TensorShape([enhancer_seq_len]),
            tf.TensorShape([annotation_size, enhancer_seq_len]),
            tf.TensorShape([enhancer_seq_len]),
            tf.TensorShape([promoter_seq_len]),
            tf.TensorShape([annotation_size, promoter_seq_len]),
            tf.TensorShape([promoter_seq_len]),
            tf.TensorShape([1]),
        ),
    )
    return ds


def parse_function(x_enhancer, annotation_enhancer, segment_enhancer,
                   x_promoter, annotation_promoter, segment_promoter, y):
    x = {
        "Input-Token_Enhancer": tf.cast(x_enhancer, tf.int32),
        "Input-Segment_Enhancer": tf.cast(segment_enhancer, tf.int32),
        "Input-Token_Promoter": tf.cast(x_promoter, tf.int32),
        "Input-Segment_Promoter": tf.cast(segment_promoter, tf.int32),
    }

    for ii in range(annotation_enhancer.shape[1]):
        x[f"Input-Token-Type_Enhancer_{ii}"] = tf.cast(annotation_enhancer[:, ii, :], tf.int32)
    for ii in range(annotation_promoter.shape[1]):
        x[f"Input-Token-Type_Promoter_{ii}"] = tf.cast(annotation_promoter[:, ii, :], tf.int32)

    y = tf.cast(y, tf.float32)
    return x, y


def build_input_dict(X_enh, A_enh, X_pro, A_pro):
    x = {
        "Input-Token_Enhancer": X_enh.astype(np.int32),
        "Input-Segment_Enhancer": np.zeros_like(X_enh, dtype=np.int32),
        "Input-Token_Promoter": X_pro.astype(np.int32),
        "Input-Segment_Promoter": np.zeros_like(X_pro, dtype=np.int32),
    }
    for ii in range(A_enh.shape[1]):
        x[f"Input-Token-Type_Enhancer_{ii}"] = A_enh[:, ii, :].astype(np.int32)
    for ii in range(A_pro.shape[1]):
        x[f"Input-Token-Type_Promoter_{ii}"] = A_pro[:, ii, :].astype(np.int32)
    return x


def model_def(vocab_size, annotation_size, drop_rate=0.25):
    config = {
        "attention_probs_dropout_prob": 0,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0,
        "embedding_size": 256,
        "hidden_size": 256,
        "initializer_range": 0.02,
        "intermediate_size": 512,
        "max_position_embeddings": 2048,
        "num_attention_heads": 8,
        "num_hidden_layers": 2,
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
        "multi_inputs": [2] * annotation_size,
    }

    bert_enhancer = build_transformer_model(configs=config, model="multi_inputs_bert", return_keras_model=False)
    bert_promoter = build_transformer_model(configs=config, model="multi_inputs_bert", return_keras_model=False)

    x_enh = tf.keras.layers.Input(shape=(None,), name="Input-Token_Enhancer")
    s_enh = tf.keras.layers.Input(shape=(None,), name="Input-Segment_Enhancer")
    x_pro = tf.keras.layers.Input(shape=(None,), name="Input-Token_Promoter")
    s_pro = tf.keras.layers.Input(shape=(None,), name="Input-Segment_Promoter")

    inputs = [x_enh, s_enh, x_pro, s_pro]
    enhancer_inputs = [x_enh, s_enh]
    promoter_inputs = [x_pro, s_pro]

    for ii in range(annotation_size):
        inp = tf.keras.layers.Input(shape=(None,), name=f"Input-Token-Type_Enhancer_{ii}")
        enhancer_inputs.append(inp)
        inputs.append(inp)
    for ii in range(annotation_size):
        inp = tf.keras.layers.Input(shape=(None,), name=f"Input-Token-Type_Promoter_{ii}")
        promoter_inputs.append(inp)
        inputs.append(inp)

    bert_enhancer.set_inputs(enhancer_inputs)
    bert_promoter.set_inputs(promoter_inputs)

    enhancer_output = bert_enhancer.model(enhancer_inputs)
    promoter_output = bert_promoter.model(promoter_inputs)

    enhancer_output = tf.keras.layers.Lambda(lambda x: x[:, 0])(enhancer_output)
    promoter_output = tf.keras.layers.Lambda(lambda x: x[:, 0])(promoter_output)

    x = tf.keras.layers.concatenate([promoter_output, enhancer_output])
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(drop_rate)(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="CLS-Activation")(x)

    return tf.keras.models.Model(inputs, output)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--cell", default="tB")
    p.add_argument("--type", default="P-E")
    p.add_argument("--frac", default="frac10_clean")
    p.add_argument("--ngram", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--max-folds", type=int, default=2)
    p.add_argument("--drop-rate", type=float, default=0.25)
    args = p.parse_args()

    annotation_size = 13
    ngram = args.ngram
    word_dict = get_word_dict_for_n_gram_number(n_gram=ngram)
    vocab_size = len(word_dict) + 10

    data_path = os.path.join(DATA_ROOT, args.cell, args.type, f"{ngram}_gram_oldcompare_13anno_frac")
    enhancer_file = os.path.join(data_path, f"enhancer_Seq_{ngram}_gram_knowledge_{args.frac}.npz")
    promoter_file = os.path.join(data_path, f"promoter_Seq_{ngram}_gram_knowledge_{args.frac}.npz")

    X_enh, A_enh, Y1 = load_all_data([enhancer_file])
    X_pro, A_pro, Y2 = load_all_data([promoter_file])
    if len(Y1) != len(Y2):
        raise ValueError("label mismatch")
    Y = Y1

    seed = 7
    np.random.seed(seed)
    tf.random.set_seed(seed)

    gpus = tf.config.list_physical_devices("GPU")
    strategy = tf.distribute.MirroredStrategy() if len(gpus) >= 2 else tf.distribute.get_strategy()
    GLOBAL_BATCH_SIZE = args.batch_size * max(1, strategy.num_replicas_in_sync)

    enhancer_seq_len = X_enh.shape[1]
    promoter_seq_len = X_pro.shape[1]

    task_name = f"{args.cell}_{args.type}_{args.frac}_oldcompare13"
    task_out_dir = os.path.join(CHECKPOINT_ROOT, _safe_name(task_name))
    _ensure_dir(task_out_dir)
    progress_file = os.path.join(METRICS_ROOT, f"progress_{_safe_name(task_name)}.json")
    metric_file = os.path.join(METRICS_ROOT, f"metrics_{_safe_name(task_name)}.json")

    progress = _load_json_or_default(progress_file, {"completed_folds": [], "fold_results": {}})
    completed = set(progress.get("completed_folds", []))
    fold_results = progress.get("fold_results", {})

    metrics_all = _load_json_or_default(metric_file, {"task_name": task_name, "updated_at": _now(), "folds": {}})

    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    for fold, (train_idx, test_idx) in tqdm(
        enumerate(kfold.split(np.arange(len(Y)), Y)),
        total=10,
        desc=f"KFold({task_name})"
    ):
        if fold >= args.max_folds:
            break

        ckpt_path = os.path.join(task_out_dir, f"best_model_fold{fold}.h5")
        if fold in completed and os.path.exists(ckpt_path):
            continue

        tf.keras.backend.clear_session()

        x_enh_train = X_enh[train_idx]
        a_enh_train = A_enh[train_idx]
        x_pro_train = X_pro[train_idx]
        a_pro_train = A_pro[train_idx]
        y_train = Y[train_idx]

        x_enh_test = X_enh[test_idx]
        a_enh_test = A_enh[test_idx]
        x_pro_test = X_pro[test_idx]
        a_pro_test = A_pro[test_idx]
        y_test = Y[test_idx]

        idx = np.arange(len(y_train))
        np.random.shuffle(idx)
        split = int(len(idx) * 0.9)
        tr_idx = idx[:split]
        va_idx = idx[split:]

        x_enh_valid = x_enh_train[va_idx]
        a_enh_valid = a_enh_train[va_idx]
        x_pro_valid = x_pro_train[va_idx]
        a_pro_valid = a_pro_train[va_idx]
        y_valid = y_train[va_idx]

        x_enh_train2 = x_enh_train[tr_idx]
        a_enh_train2 = a_enh_train[tr_idx]
        x_pro_train2 = x_pro_train[tr_idx]
        a_pro_train2 = a_pro_train[tr_idx]
        y_train2 = y_train[tr_idx]

        train_steps = max(1, len(y_train2) // GLOBAL_BATCH_SIZE)
        valid_steps = max(1, len(y_valid) // GLOBAL_BATCH_SIZE)
        test_steps = max(1, len(y_test) // GLOBAL_BATCH_SIZE)

        callbacks = [
            ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True, verbose=0),
            EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-6, verbose=0),
        ]

        train_ds = make_tf_dataset(x_enh_train2, a_enh_train2, x_pro_train2, a_pro_train2, y_train2,
                                   enhancer_seq_len, promoter_seq_len, annotation_size, shuffle=True)
        train_ds = train_ds.batch(args.batch_size).map(parse_function, num_parallel_calls=4).prefetch(tf.data.experimental.AUTOTUNE)

        valid_ds = make_tf_dataset(x_enh_valid, a_enh_valid, x_pro_valid, a_pro_valid, y_valid,
                                   enhancer_seq_len, promoter_seq_len, annotation_size, shuffle=False)
        valid_ds = valid_ds.batch(args.batch_size).map(parse_function, num_parallel_calls=4).prefetch(tf.data.experimental.AUTOTUNE)

        test_ds = make_tf_dataset(x_enh_test, a_enh_test, x_pro_test, a_pro_test, y_test,
                                  enhancer_seq_len, promoter_seq_len, annotation_size, shuffle=False)
        test_ds = test_ds.batch(args.batch_size).map(parse_function, num_parallel_calls=4).prefetch(tf.data.experimental.AUTOTUNE)

        with strategy.scope():
            model = model_def(vocab_size=vocab_size, annotation_size=annotation_size, drop_rate=args.drop_rate)
            model.compile(
                loss="binary_crossentropy",
                optimizer=Adam(learning_rate=1e-4),
                metrics=["acc", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall"), f1_score],
            )

        model.fit(
            train_ds,
            steps_per_epoch=train_steps,
            epochs=args.epochs,
            validation_data=valid_ds,
            validation_steps=valid_steps,
            callbacks=callbacks,
            verbose=1,
        )

        with strategy.scope():
            model2 = model_def(vocab_size=vocab_size, annotation_size=annotation_size, drop_rate=args.drop_rate)
            model2.compile(
                loss="binary_crossentropy",
                optimizer=Adam(learning_rate=1e-4),
                metrics=["acc", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall"), f1_score],
            )
            model2.load_weights(ckpt_path)

        eval_res = model2.evaluate(test_ds, steps=test_steps, verbose=1)
        eval_list = [float(x) for x in eval_res]

        x_test_dict = build_input_dict(x_enh_test, a_enh_test, x_pro_test, a_pro_test)
        y_score = model2.predict(x_test_dict, batch_size=args.batch_size, verbose=0).reshape(-1)
        y_true = y_test.reshape(-1).astype(np.int32)

        try:
            auc_v = float(roc_auc_score(y_true, y_score))
        except Exception:
            auc_v = None
        try:
            auprc_v = float(average_precision_score(y_true, y_score))
        except Exception:
            auprc_v = None

        fold_key = f"fold_{fold}"
        fold_results[fold_key] = {
            "time": _now(),
            "frac": args.frac,
            "annotation_size": annotation_size,
            "steps": {"train_steps": int(train_steps), "valid_steps": int(valid_steps), "test_steps": int(test_steps)},
            "metrics_truncated_test": {
                "loss": eval_list[0],
                "acc": eval_list[1],
                "precision": eval_list[2],
                "recall": eval_list[3],
                "f1": eval_list[4],
                "raw_list": eval_list,
            },
            "metrics_full_test_extra": {"auc": auc_v, "auprc": auprc_v},
        }

        completed.add(fold)
        progress["completed_folds"] = sorted(completed)
        progress["fold_results"] = fold_results
        _save_json_atomic(progress_file, progress)

        metrics_all["updated_at"] = _now()
        metrics_all["folds"][fold_key] = fold_results[fold_key]
        _save_json_atomic(metric_file, metrics_all)

        print(f"[Fold Done] fold={fold} frac={args.frac} truncated_test={eval_list} full_auc={auc_v} full_auprc={auprc_v}")

        del model, model2, train_ds, valid_ds, test_ds, x_test_dict, y_score, y_true
        gc.collect()
        tf.keras.backend.clear_session()

    print("[DONE]")


if __name__ == "__main__":
    main()
