# -*- coding:utf-8 -*-
import os
import sys
import gc
import json
import time
import math
import argparse
import warnings
import logging
import subprocess
from datetime import datetime

import numpy as np
from tqdm import tqdm

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)


def _maybe_install_biopython():
    try:
        import Bio  # noqa
        return
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "biopython"])


def _maybe_install_sklearn():
    try:
        import sklearn  # noqa
        return
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "scikit-learn"])


_maybe_install_biopython()
_maybe_install_sklearn()

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
    absl_logging.set_stderrthreshold("error")
except Exception:
    pass

from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Lambda, concatenate
import tensorflow.keras.backend as K

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

PROJECT_ROOT = "/content/drive/MyDrive/LOGO_project/test"
DATA_ROOT = "/content/drive/MyDrive/LOGO_project/test/03_LOGO_EPI/data/data"

CHECKPOINT_ROOT = "/content/drive/MyDrive/LOGO_outputs/checkpoints_epi_clean"
METRICS_ROOT = "/content/drive/MyDrive/LOGO_outputs/metrics_epi_clean"

PRETRAIN_PATH = "/content/drive/MyDrive/LOGO_project/test/99_PreTrain_Model_Weight/LOGO_5_gram_2_layer_8_heads_256_dim_weights_32-0.885107.hdf5"

sys.path.insert(0, PROJECT_ROOT)
from bgi.bert4keras.models import build_transformer_model
from bgi.common.refseq_utils import get_word_dict_for_n_gram_number

DEBUG = False

include_types = [
    "enhancer",
    "promoter",
    "pseudogene",
    "insulator",
    "conserved_region",
    "protein_binding_site",
    "DNAseI_hypersensitive_site",
    "nucleotide_cleavage_site",
    "silencer",
    "gene",
    "exon",
    "CDS",
    "TATA_box",
]
exclude_annotation = []


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _safe_name(s: str) -> str:
    return s.replace("/", "_").replace("\\", "_").replace(" ", "_")


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _progress_path(task_name: str, ngram: int, annotation_size: int, frac_tag: str) -> str:
    return os.path.join(
        METRICS_ROOT,
        "progress",
        frac_tag,
        f"progress_{_safe_name(task_name)}_{ngram}gram_{annotation_size}anno_{frac_tag}.json"
    )


def _metrics_path(task_name: str, ngram: int, annotation_size: int, frac_tag: str) -> str:
    return os.path.join(
        METRICS_ROOT,
        "metrics",
        frac_tag,
        f"metrics_{_safe_name(task_name)}_{ngram}gram_{annotation_size}anno_{frac_tag}.json"
    )


def _pred_dir(task_name: str, frac_tag: str) -> str:
    return os.path.join(METRICS_ROOT, "metrics", frac_tag, _safe_name(task_name), "predictions")


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


class ComputeMonitor(tf.keras.callbacks.Callback):
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


def f1_score(y_true, y_pred):
    y_true = tf.cast(y_true, "float32")
    y_pred = tf.cast(y_pred, "float32")
    tp = K.sum(tf.cast(K.equal(y_true, 1) & K.equal(K.round(y_pred), 1), "float32"))
    fp = K.sum(tf.cast(K.equal(y_true, 0) & K.equal(K.round(y_pred), 1), "float32"))
    fn = K.sum(tf.cast(K.equal(y_true, 1) & K.equal(K.round(y_pred), 0), "float32"))
    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())
    return 2 * p * r / (p + r + K.epsilon())


def weighted_bce(pos_weight):
    pos_weight = tf.constant(float(pos_weight), dtype=tf.float32)

    def loss(y_true, y_pred):
        y_true_f = tf.cast(y_true, tf.float32)
        y_pred_f = tf.cast(y_pred, tf.float32)
        y_pred_f = tf.clip_by_value(y_pred_f, K.epsilon(), 1.0 - K.epsilon())

        loss_pos = - y_true_f * tf.math.log(y_pred_f) * pos_weight
        loss_neg = - (1.0 - y_true_f) * tf.math.log(1.0 - y_pred_f)
        return tf.reduce_mean(loss_pos + loss_neg)

    return loss


def load_npz_data_for_classification(file_name, ngram=5, only_one_slice=True, ngram_index=1):
    if (not str(file_name).endswith(".npz")) or (not os.path.exists(file_name)):
        raise FileNotFoundError(f"NPZ not found: {os.path.abspath(file_name)}")

    loaded = np.load(file_name, allow_pickle=True)
    keys = loaded.files

    if "sequence" in keys:
        x_data = loaded["sequence"]
    elif "x" in keys:
        x_data = loaded["x"]
    else:
        raise KeyError(f"No sequence/x key found in {file_name}. keys={keys}")

    if "label" in keys:
        y_data = loaded["label"]
    elif "y" in keys:
        y_data = loaded["y"]
    else:
        raise KeyError(f"No label/y key found in {file_name}. keys={keys}")

    if "annotation" in keys:
        anno_data = loaded["annotation"]
    else:
        anno_data = np.zeros((len(x_data), len(include_types), x_data.shape[1]), dtype=np.int8)

    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)
    anno_data = np.asarray(anno_data)

    num_features = len(include_types)
    if anno_data.ndim != 3:
        raise ValueError(
            f"annotation ndim must be 3, got {anno_data.ndim}, "
            f"file={file_name}, shape={anno_data.shape}"
        )

    if anno_data.shape[1] < num_features:
        pad = np.zeros(
            (anno_data.shape[0], num_features - anno_data.shape[1], anno_data.shape[2]),
            dtype=anno_data.dtype
        )
        anno_data = np.concatenate([anno_data, pad], axis=1)
    elif anno_data.shape[1] > num_features:
        anno_data = anno_data[:, :num_features, :]

    if DEBUG:
        print(f"[Load] {file_name}")
        print("keys:", keys)
        print("x_data:", x_data.shape)
        print("anno_data:", anno_data.shape)
        print("y_data:", y_data.shape)

    x_all, anno_all, y_all = [], [], []
    if only_one_slice:
        max_slice_seq_len = x_data.shape[1] // ngram * ngram
        for ii in range(ngram):
            if ngram_index is not None and ii != ngram_index:
                continue
            slice_indexes = list(range(ii, max_slice_seq_len, ngram))
            x_all.append(x_data[:, slice_indexes])
            anno_all.append(anno_data[:, :, slice_indexes])
            y_all.append(y_data)
    else:
        x_all.append(x_data)
        anno_all.append(anno_data)
        y_all.append(y_data)

    return x_all, anno_all, y_all


def load_all_data(record_names, ngram=5, only_one_slice=True, ngram_index=1):
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


def make_tf_dataset(
    X_enh, A_enh,
    X_pro, A_pro,
    Y,
    enhancer_seq_len,
    promoter_seq_len,
    annotation_size,
    shuffle=False,
    repeat=True,
):
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

            if not repeat:
                break

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
        gene_type = annotation_enhancer[:, ii, :]
        if ii in exclude_annotation:
            gene_type = tf.zeros_like(gene_type, dtype=tf.int32)
        x[f"Input-Token-Type_Enhancer_{ii}"] = tf.cast(gene_type, tf.int32)

    for ii in range(annotation_promoter.shape[1]):
        gene_type = annotation_promoter[:, ii, :]
        if ii in exclude_annotation:
            gene_type = tf.zeros_like(gene_type, dtype=tf.int32)
        x[f"Input-Token-Type_Promoter_{ii}"] = tf.cast(gene_type, tf.int32)

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
        gene_type = A_enh[:, ii, :]
        if ii in exclude_annotation:
            gene_type = np.zeros_like(gene_type, dtype=np.int32)
        x[f"Input-Token-Type_Enhancer_{ii}"] = gene_type.astype(np.int32)

    for ii in range(A_pro.shape[1]):
        gene_type = A_pro[:, ii, :]
        if ii in exclude_annotation:
            gene_type = np.zeros_like(gene_type, dtype=np.int32)
        x[f"Input-Token-Type_Promoter_{ii}"] = gene_type.astype(np.int32)

    return x


def model_def(
    vocab_size,
    annotation_size,
    embedding_size=256,
    hidden_size=256,
    num_heads=8,
    num_hidden_layers=2,
    intermediate_size=512,
    max_position_embeddings=2048,
    drop_rate=0.25,
    load_pretrain=True,
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

    bert_enhancer = build_transformer_model(
        configs=config,
        model="multi_inputs_bert",
        return_keras_model=False,
    )
    bert_promoter = build_transformer_model(
        configs=config,
        model="multi_inputs_bert",
        return_keras_model=False,
    )

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

    enhancer_output = Lambda(lambda x: x[:, 0])(enhancer_output)
    promoter_output = Lambda(lambda x: x[:, 0])(promoter_output)

    x = concatenate([promoter_output, enhancer_output])
    x = BatchNormalization()(x)
    x = Dropout(drop_rate)(x)
    output = Dense(1, activation="sigmoid", name="CLS-Activation")(x)

    model = tf.keras.models.Model(inputs, output)

    if load_pretrain:
        try:
            model.load_weights(PRETRAIN_PATH, by_name=True, skip_mismatch=True)
            print("\n[INFO] ✅ 成功加载预训练权重（LOGO backbone）\n")
        except Exception as e:
            print(f"\n[WARNING] ⚠️ 预训练权重加载失败: {e}\n")

    return model


def build_compiled_model(strategy, vocab_size, annotation_size, drop_rate, learning_rate, loss_name, pos_weight, load_pretrain):
    with strategy.scope():
        model = model_def(
            vocab_size=vocab_size,
            annotation_size=annotation_size,
            drop_rate=drop_rate,
            load_pretrain=load_pretrain,
        )

        if loss_name == "weighted_bce":
            loss_fn = weighted_bce(pos_weight)
        else:
            loss_fn = "binary_crossentropy"

        model.compile(
            loss=loss_fn,
            optimizer=Adam(learning_rate=learning_rate),
            metrics=[
                "acc",
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.AUC(name="roc_auc", curve="ROC"),
                tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
                f1_score,
            ],
        )
    return model


def evaluate_and_record_fold(
    strategy,
    ckpt_path,
    fold,
    task_name,
    frac_tag,
    batch_size,
    vocab_size,
    annotation_size,
    drop_rate,
    learning_rate,
    loss_name,
    pos_weight,
    x_enh_test,
    a_enh_test,
    x_pro_test,
    a_pro_test,
    y_test,
    fold_results,
    progress,
    progress_file,
    metrics_all,
    metric_file,
):
    if not os.path.exists(ckpt_path):
        print(f"[TEST-SKIP] best checkpoint not found: {ckpt_path}", flush=True)
        return

    model2 = build_compiled_model(
        strategy=strategy,
        vocab_size=vocab_size,
        annotation_size=annotation_size,
        drop_rate=drop_rate,
        learning_rate=learning_rate,
        loss_name=loss_name,
        pos_weight=pos_weight,
        load_pretrain=False,
    )
    model2.load_weights(ckpt_path)

    x_test_dict = build_input_dict(x_enh_test, a_enh_test, x_pro_test, a_pro_test)
    y_test_2d = y_test.reshape(-1, 1).astype(np.float32)

    eval_res = model2.evaluate(
        x_test_dict,
        y_test_2d,
        batch_size=batch_size,
        verbose=1,
    )
    eval_list = _to_float_list(eval_res)

    loss_v = eval_list[0]
    acc_v = eval_list[1]
    prec_v = eval_list[2]
    rec_v = eval_list[3]
    roc_auc_batch_v = eval_list[4]
    pr_auc_batch_v = eval_list[5]
    f1_v = eval_list[6]

    y_score = model2.predict(
        x_test_dict,
        batch_size=batch_size,
        verbose=0,
    ).reshape(-1)

    y_true = y_test.reshape(-1).astype(np.int32)

    try:
        auc_v = float(roc_auc_score(y_true, y_score))
    except Exception:
        auc_v = None

    try:
        auprc_v = float(average_precision_score(y_true, y_score))
    except Exception:
        auprc_v = None

    pred_dir = _pred_dir(task_name, frac_tag)
    _ensure_dir(pred_dir)
    y_true_path = os.path.join(pred_dir, f"{_safe_name(task_name)}_{frac_tag}_fold{fold}_y_true.npy")
    y_score_path = os.path.join(pred_dir, f"{_safe_name(task_name)}_{frac_tag}_fold{fold}_y_score.npy")
    np.save(y_true_path, y_true)
    np.save(y_score_path, y_score)

    fold_key = f"fold_{fold}"
    previous = fold_results.get(fold_key, {})
    fold_results[fold_key] = {
        **previous,
        "time": _now(),
        "ckpt_path": os.path.abspath(ckpt_path),
        "prediction_paths": {
            "y_true_npy": os.path.abspath(y_true_path),
            "y_score_npy": os.path.abspath(y_score_path),
        },
        "learning_rate": learning_rate,
        "dropout": drop_rate,
        "loss_name": loss_name,
        "monitor": "val_pr_auc",
        "metrics": {
            "loss": loss_v,
            "acc": acc_v,
            "precision": prec_v,
            "recall": rec_v,
            "roc_auc_batch_metric": roc_auc_batch_v,
            "pr_auc_batch_metric": pr_auc_batch_v,
            "f1": f1_v,
            "auc": auc_v,
            "auprc": auprc_v,
            "raw_list": eval_list,
        }
    }

    progress["fold_results"] = fold_results
    _save_progress(progress_file, progress)

    metrics_all["updated_at"] = _now()
    metrics_all["folds"][fold_key] = fold_results[fold_key]
    _save_json_atomic(metric_file, metrics_all)

    print(
        f"[TEST DONE] {task_name} {frac_tag} fold {fold} | "
        f"AUC={auc_v} | AUPRC={auprc_v}",
        flush=True
    )

    del model2, x_test_dict, y_test_2d, eval_res, eval_list, y_score, y_true
    gc.collect()
    tf.keras.backend.clear_session()


def train_kfold(
    CELL,
    TYPE,
    frac_tag,
    batch_size=128,
    annotation_size=13,
    epochs=30,
    ngram=5,
    vocab_size=10000,
    task_name=None,
    early_patience=6,
    max_folds_to_run=10,
    learning_rate=2e-5,
    drop_rate=0.25,
    loss_name="weighted_bce",
    mode="train_and_test",
):
    print(
        f"[DEBUG] train_kfold entered: CELL={CELL}, TYPE={TYPE}, frac={frac_tag}, "
        f"lr={learning_rate}, drop={drop_rate}, loss={loss_name}, mode={mode}",
        flush=True
    )

    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) >= 2:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.get_strategy()

    GLOBAL_BATCH_SIZE = batch_size * max(1, strategy.num_replicas_in_sync)
    VAL_BATCH_SIZE = batch_size * 2
    num_parallel_calls = 4
    only_one_slice = True

    data_path = os.path.join(DATA_ROOT, CELL, TYPE, f"{ngram}_gram")

    enhancer_file_candidates = [
        os.path.join(data_path, f"enhancer_Seq_{ngram}_gram_knowledge_{frac_tag}.npz"),
        os.path.join(data_path, f"enhancer_Seq_{ngram}_gram_{frac_tag}.npz"),
    ]
    promoter_file_candidates = [
        os.path.join(data_path, f"promoter_Seq_{ngram}_gram_knowledge_{frac_tag}.npz"),
        os.path.join(data_path, f"promoter_Seq_{ngram}_gram_{frac_tag}.npz"),
    ]

    enhancer_file = next((p for p in enhancer_file_candidates if os.path.exists(p)), None)
    promoter_file = next((p for p in promoter_file_candidates if os.path.exists(p)), None)

    if enhancer_file is None:
        raise FileNotFoundError(f"No enhancer file found in: {enhancer_file_candidates}")
    if promoter_file is None:
        raise FileNotFoundError(f"No promoter file found in: {promoter_file_candidates}")

    print("[DEBUG] enhancer_file =", enhancer_file, flush=True)
    print("[DEBUG] promoter_file =", promoter_file, flush=True)

    X_enh, A_enh, Y1 = load_all_data([enhancer_file], ngram=ngram, only_one_slice=only_one_slice, ngram_index=1)
    X_pro, A_pro, Y2 = load_all_data([promoter_file], ngram=ngram, only_one_slice=only_one_slice, ngram_index=1)

    if len(Y1) != len(Y2):
        raise ValueError(f"Enhancer/Promoter label length mismatch: {len(Y1)} vs {len(Y2)}")

    Y = Y1
    enhancer_seq_len = X_enh.shape[1]
    promoter_seq_len = X_pro.shape[1]

    seed = 7
    np.random.seed(seed)
    tf.random.set_seed(seed)

    if task_name is None:
        task_name = f"{CELL}_{TYPE}_EPI"

    _ensure_dir(CHECKPOINT_ROOT)
    _ensure_dir(METRICS_ROOT)

    task_out_dir = os.path.join(CHECKPOINT_ROOT, frac_tag, _safe_name(task_name))
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
        "cell": CELL,
        "type": TYPE,
        "frac_tag": frac_tag,
        "ngram": ngram,
        "annotation_size": annotation_size,
        "learning_rate": learning_rate,
        "dropout": drop_rate,
        "loss_name": loss_name,
        "monitor": "val_pr_auc",
        "folds": {}
    })

    X_index = np.arange(len(Y))
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    for fold, (train_idx, test_idx) in tqdm(
        enumerate(kfold.split(X_index, Y)),
        total=10,
        desc=f"KFold({task_name},{frac_tag})",
        dynamic_ncols=True,
        leave=True,
    ):
        if fold >= max_folds_to_run:
            print(f"[STOP] 已达到 max_folds_to_run={max_folds_to_run}，停止后续 folds。", flush=True)
            break

        ckpt_path = os.path.join(task_out_dir, f"best_model_{_safe_name(task_name)}_{frac_tag}_fold{fold}.h5")

        tf.keras.backend.clear_session()

        tr_ids, va_ids = train_test_split(
            train_idx,
            test_size=0.1,
            random_state=seed + fold,
            stratify=Y[train_idx]
        )

        x_enh_train, a_enh_train = X_enh[tr_ids], A_enh[tr_ids]
        x_pro_train, a_pro_train = X_pro[tr_ids], A_pro[tr_ids]
        y_train = Y[tr_ids]

        x_enh_valid, a_enh_valid = X_enh[va_ids], A_enh[va_ids]
        x_pro_valid, a_pro_valid = X_pro[va_ids], A_pro[va_ids]
        y_valid = Y[va_ids]

        x_enh_test, a_enh_test = X_enh[test_idx], A_enh[test_idx]
        x_pro_test, a_pro_test = X_pro[test_idx], A_pro[test_idx]
        y_test = Y[test_idx]

        n_pos = int(np.sum(y_train == 1))
        n_neg = int(np.sum(y_train == 0))
        pos_weight = float(n_neg / max(n_pos, 1))

        train_steps = max(1, math.ceil(len(y_train) / GLOBAL_BATCH_SIZE))
        validation_steps = max(1, math.ceil(len(y_valid) / VAL_BATCH_SIZE))

        if mode == "test_only":
            evaluate_and_record_fold(
                strategy=strategy,
                ckpt_path=ckpt_path,
                fold=fold,
                task_name=task_name,
                frac_tag=frac_tag,
                batch_size=batch_size,
                vocab_size=vocab_size,
                annotation_size=annotation_size,
                drop_rate=drop_rate,
                learning_rate=learning_rate,
                loss_name=loss_name,
                pos_weight=pos_weight,
                x_enh_test=x_enh_test,
                a_enh_test=a_enh_test,
                x_pro_test=x_pro_test,
                a_pro_test=a_pro_test,
                y_test=y_test,
                fold_results=fold_results,
                progress=progress,
                progress_file=progress_file,
                metrics_all=metrics_all,
                metric_file=metric_file,
            )

            del x_enh_train, a_enh_train, x_pro_train, a_pro_train, y_train
            del x_enh_valid, a_enh_valid, x_pro_valid, a_pro_valid, y_valid
            del x_enh_test, a_enh_test, x_pro_test, a_pro_test, y_test
            gc.collect()
            tf.keras.backend.clear_session()
            continue

        if fold in completed and os.path.exists(ckpt_path):
            print(f"[SKIP-TRAIN] fold {fold} 已完成，跳过训练。", flush=True)
            if mode == "train_and_test":
                evaluate_and_record_fold(
                    strategy=strategy,
                    ckpt_path=ckpt_path,
                    fold=fold,
                    task_name=task_name,
                    frac_tag=frac_tag,
                    batch_size=batch_size,
                    vocab_size=vocab_size,
                    annotation_size=annotation_size,
                    drop_rate=drop_rate,
                    learning_rate=learning_rate,
                    loss_name=loss_name,
                    pos_weight=pos_weight,
                    x_enh_test=x_enh_test,
                    a_enh_test=a_enh_test,
                    x_pro_test=x_pro_test,
                    a_pro_test=a_pro_test,
                    y_test=y_test,
                    fold_results=fold_results,
                    progress=progress,
                    progress_file=progress_file,
                    metrics_all=metrics_all,
                    metric_file=metric_file,
                )

            del x_enh_train, a_enh_train, x_pro_train, a_pro_train, y_train
            del x_enh_valid, a_enh_valid, x_pro_valid, a_pro_valid, y_valid
            del x_enh_test, a_enh_test, x_pro_test, a_pro_test, y_test
            gc.collect()
            tf.keras.backend.clear_session()
            continue

        compute_cb = ComputeMonitor()

        callbacks = [
            compute_cb,
            ModelCheckpoint(
                ckpt_path,
                monitor="val_pr_auc",
                mode="max",
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor="val_pr_auc",
                mode="max",
                patience=early_patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor="val_pr_auc",
                mode="max",
                factor=0.5,
                patience=2,
                min_lr=1e-6,
                verbose=1
            ),
        ]

        train_ds = make_tf_dataset(
            x_enh_train, a_enh_train, x_pro_train, a_pro_train, y_train,
            enhancer_seq_len, promoter_seq_len, annotation_size,
            shuffle=True, repeat=True
        )
        train_ds = train_ds.batch(batch_size)
        train_ds = train_ds.map(parse_function, num_parallel_calls=num_parallel_calls)
        train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

        valid_ds = make_tf_dataset(
            x_enh_valid, a_enh_valid, x_pro_valid, a_pro_valid, y_valid,
            enhancer_seq_len, promoter_seq_len, annotation_size,
            shuffle=False, repeat=False
        )
        valid_ds = valid_ds.batch(VAL_BATCH_SIZE)
        valid_ds = valid_ds.map(parse_function, num_parallel_calls=num_parallel_calls)
        valid_ds = valid_ds.prefetch(tf.data.experimental.AUTOTUNE)

        model = build_compiled_model(
            strategy=strategy,
            vocab_size=vocab_size,
            annotation_size=annotation_size,
            drop_rate=drop_rate,
            learning_rate=learning_rate,
            loss_name=loss_name,
            pos_weight=pos_weight,
            load_pretrain=True,
        )

        model.fit(
            train_ds,
            steps_per_epoch=train_steps,
            epochs=epochs,
            validation_data=valid_ds,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1,
        )

        completed.add(fold)
        progress["completed_folds"] = sorted(completed)
        progress["fold_results"] = fold_results
        _save_progress(progress_file, progress)

        if mode == "train_and_test":
            evaluate_and_record_fold(
                strategy=strategy,
                ckpt_path=ckpt_path,
                fold=fold,
                task_name=task_name,
                frac_tag=frac_tag,
                batch_size=batch_size,
                vocab_size=vocab_size,
                annotation_size=annotation_size,
                drop_rate=drop_rate,
                learning_rate=learning_rate,
                loss_name=loss_name,
                pos_weight=pos_weight,
                x_enh_test=x_enh_test,
                a_enh_test=a_enh_test,
                x_pro_test=x_pro_test,
                a_pro_test=a_pro_test,
                y_test=y_test,
                fold_results=fold_results,
                progress=progress,
                progress_file=progress_file,
                metrics_all=metrics_all,
                metric_file=metric_file,
            )
            print(f"[Fold Done] {task_name} {frac_tag} fold {fold}", flush=True)
        else:
            print(f"[Fold Train Done] {task_name} {frac_tag} fold {fold}", flush=True)

        del model
        del train_ds, valid_ds
        del x_enh_train, a_enh_train, x_pro_train, a_pro_train, y_train
        del x_enh_valid, a_enh_valid, x_pro_valid, a_pro_valid, y_valid
        del x_enh_test, a_enh_test, x_pro_test, a_pro_test, y_test
        gc.collect()
        tf.keras.backend.clear_session()

    del X_enh, A_enh, X_pro, A_pro, Y
    gc.collect()
    tf.keras.backend.clear_session()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fracs", nargs="+", default=["frac10_clean", "frac20_clean", "frac50_clean"])
    parser.add_argument("--drop", type=float, default=0.25)
    parser.add_argument("--max-folds", type=int, default=10)
    parser.add_argument("--cells", nargs="+", default=["tB"])
    parser.add_argument("--type", default="P-E")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--loss", default="weighted_bce")
    parser.add_argument("--ngram", type=int, default=5)
    parser.add_argument("--annotation-size", type=int, default=13)
    parser.add_argument(
        "--mode",
        choices=["train_and_test", "train_only", "test_only"],
        default="train_and_test"
    )
    return parser.parse_args()


if __name__ == "__main__":
    print("[READY] Full logic loaded. Starting Training...", flush=True)
    print("[TIP] train+test: !python 04_LOGO_EPI_train_conv1d_concat_atcg.py --fracs frac10_clean --drop 0.25 --max-folds 1", flush=True)
    print("[TIP] train only:  !python 04_LOGO_EPI_train_conv1d_concat_atcg.py --fracs frac10_clean --drop 0.25 --max-folds 1 --mode train_only", flush=True)
    print("[TIP] test only:   !python 04_LOGO_EPI_train_conv1d_concat_atcg.py --fracs frac10_clean --drop 0.25 --max-folds 1 --mode test_only", flush=True)

    args = parse_args()

    epi_dir = os.path.join(PROJECT_ROOT, "03_LOGO_EPI")
    os.chdir(epi_dir)
    print("[CWD]", os.getcwd(), flush=True)

    ngram = args.ngram
    word_dict = get_word_dict_for_n_gram_number(n_gram=ngram)
    vocab_size = len(word_dict) + 10

    annotation_size = args.annotation_size
    TYPE = args.type
    CELLs = args.cells

    FRAC_TAGS = args.fracs
    batch_size = args.batch_size
    epochs = args.epochs
    LEARNING_RATE = args.lr
    DROP_RATE = args.drop
    LOSS_NAME = args.loss
    MAX_FOLDS = args.max_folds
    MODE = args.mode

    print(f"[RUN CONFIG] FRAC_TAGS={FRAC_TAGS}", flush=True)
    print(f"[RUN CONFIG] DROP_RATE={DROP_RATE}", flush=True)
    print(f"[RUN CONFIG] MAX_FOLDS={MAX_FOLDS}", flush=True)
    print(f"[RUN CONFIG] MODE={MODE}", flush=True)
    print(f"[RUN CONFIG] batch_size={batch_size}, VAL_BATCH_SIZE={batch_size * 2}", flush=True)

    for frac_tag in FRAC_TAGS:
        print(
            f"\n================ Experiment: frac={frac_tag}, "
            f"lr={LEARNING_RATE}, drop={DROP_RATE}, loss={LOSS_NAME}, mode={MODE} ================\n",
            flush=True
        )

        for CELL in CELLs:
            drop_tag = str(DROP_RATE).replace(".", "")
            task_name = (
                f"{CELL}_{TYPE}_EPI_"
                f"lr{LEARNING_RATE:.0e}_"
                f"drop{drop_tag}_"
                f"{LOSS_NAME}_"
                f"selectedv3clean"
            )

            train_kfold(
                CELL=CELL,
                TYPE=TYPE,
                frac_tag=frac_tag,
                batch_size=batch_size,
                epochs=epochs,
                vocab_size=vocab_size,
                annotation_size=annotation_size,
                ngram=ngram,
                task_name=task_name,
                early_patience=6,
                max_folds_to_run=MAX_FOLDS,
                learning_rate=LEARNING_RATE,
                drop_rate=DROP_RATE,
                loss_name=LOSS_NAME,
                mode=MODE,
            )

    print("[DONE] Script finished normally.", flush=True)