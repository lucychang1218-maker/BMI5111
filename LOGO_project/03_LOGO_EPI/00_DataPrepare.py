# -*- coding: utf-8 -*-
import os
import sys
import random
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# =========================================================
# 0) Args
# =========================================================
CELL = sys.argv[1] if len(sys.argv) > 1 else "tB"
TYPE = sys.argv[2] if len(sys.argv) > 2 else "P-E"
TASK = sys.argv[3] if len(sys.argv) > 3 else "train"   # train / test

# =========================================================
# 1) Paths
# =========================================================
PROJECT_ROOT = "/content/drive/MyDrive/LOGO_project/test/03_LOGO_EPI"
BASE_PATH = os.path.join(PROJECT_ROOT, "data", "data", CELL, TYPE)

REF_FASTA = "/content/drive/MyDrive/LOGO_project/test/03_LOGO_EPI/data/hg19/GCF_000001405.25_GRCh37.p13_genomic.fna"
REF_GFF = "/content/drive/MyDrive/LOGO_project/test/03_LOGO_EPI/data/hg19/GCF_000001405.25_GRCh37.p13_genomic.gff"

sys.path.insert(0, "/content/drive/MyDrive/LOGO_project/test")
from bgi.common.genebank_utils import get_refseq_gff, get_gene_feature_array

# =========================================================
# 2) Config
# =========================================================
RANDOM_STATE = 42
CHUNK_SIZE = 5000

ENHANCER_LEN = 2000
PROMOTER_LEN = 1000

STRICT_FILTER = True

# 正类增强倍数：每个正类样本总共生成 5 个版本
POS_AUG_TIMES = 5

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
gene_type_dict = {k: i + 1 for i, k in enumerate(include_types)}
ANNOTATION_SIZE = len(include_types)

CHR_TO_NC = {
    "chr1": "NC_000001.10",
    "chr2": "NC_000002.11",
    "chr3": "NC_000003.11",
    "chr4": "NC_000004.11",
    "chr5": "NC_000005.9",
    "chr6": "NC_000006.11",
    "chr7": "NC_000007.13",
    "chr8": "NC_000008.10",
    "chr9": "NC_000009.11",
    "chr10": "NC_000010.10",
    "chr11": "NC_000011.9",
    "chr12": "NC_000012.11",
    "chr13": "NC_000013.10",
    "chr14": "NC_000014.8",
    "chr15": "NC_000015.9",
    "chr16": "NC_000016.9",
    "chr17": "NC_000017.10",
    "chr18": "NC_000018.9",
    "chr19": "NC_000019.9",
    "chr20": "NC_000020.10",
    "chr21": "NC_000021.8",
    "chr22": "NC_000022.10",
    "chrX": "NC_000023.10",
    "chrY": "NC_000024.9",
    "chrM": "NC_012920.1",
}

# =========================================================
# 3) DNA helpers
# =========================================================
def is_valid_dna_strict(seq: str) -> bool:
    if not isinstance(seq, str) or len(seq) == 0:
        return False
    seq = seq.upper()
    return all(ch in {"A", "C", "G", "T"} for ch in seq)


def one_hot(seq: str, strict_filter=True) -> np.ndarray:
    seq = str(seq).upper()
    if strict_filter and not is_valid_dna_strict(seq):
        return np.array([], dtype=np.int8)

    mapping = {
        "A": [1, 0, 0, 0],
        "C": [0, 1, 0, 0],
        "G": [0, 0, 1, 0],
        "T": [0, 0, 0, 1],
    }

    arr = []
    for ch in seq:
        if ch not in mapping:
            return np.array([], dtype=np.int8)
        arr.extend(mapping[ch])

    return np.array(arr, dtype=np.int8)


def normalize_chr(chrom: str) -> str:
    chrom = str(chrom)
    if chrom.startswith("chr"):
        return chrom
    return f"chr{chrom}"


def map_chr_to_fasta_key(chrom: str, sequence_dict: dict) -> str:
    chrom = normalize_chr(chrom)
    if chrom in sequence_dict:
        return chrom
    if chrom in CHR_TO_NC and CHR_TO_NC[chrom] in sequence_dict:
        return CHR_TO_NC[chrom]
    return chrom


def clamp_window(seq_len: int, start: int, end: int, target_len: int) -> Tuple[int, int]:
    start = int(start)
    end = int(end)

    if start < 0:
        end += -start
        start = 0
    if end > seq_len:
        shift = end - seq_len
        start -= shift
        end = seq_len
    if start < 0:
        start = 0

    cur_len = end - start

    if cur_len > target_len:
        mid = (start + end) // 2
        start = mid - target_len // 2
        end = start + target_len
        if start < 0:
            start = 0
            end = target_len
        if end > seq_len:
            end = seq_len
            start = end - target_len

    elif cur_len < target_len:
        need = target_len - cur_len
        left = need // 2
        right = need - left
        start = max(0, start - left)
        end = min(seq_len, end + right)
        if end - start < target_len:
            if start == 0:
                end = min(seq_len, target_len)
            elif end == seq_len:
                start = max(0, seq_len - target_len)

    return int(start), int(end)


def random_resize_window(start: int, end: int, target_len: int) -> Tuple[int, int]:
    """
    给正类做窗口增强：
    保证原 region 被包含在 target_len 的窗口内，但左右边界随机变化
    """
    start = int(start)
    end = int(end)
    region_len = max(1, end - start)

    if region_len >= target_len:
        mid = (start + end) // 2
        new_start = mid - target_len // 2
        new_end = new_start + target_len
        return new_start, new_end

    slack = target_len - region_len
    left_extra = random.randint(0, slack)
    right_extra = slack - left_extra
    return start - left_extra, end + right_extra


# =========================================================
# 4) Annotation helpers
# =========================================================
def get_annotation_matrix(gene_features, chr_name: str, start: int, end: int, seq_len: int) -> np.ndarray:
    anno = np.zeros((ANNOTATION_SIZE, seq_len), dtype=np.int8)

    feature_list = get_gene_feature_array(
        gene_features,
        chr_name,
        start,
        end
    )

    if feature_list is None or len(feature_list) == 0:
        return anno

    for item in feature_list:
        try:
            gene_type = item[0]
            rel_s = int(item[1])
            rel_e = int(item[2])
        except Exception:
            continue

        if gene_type not in gene_type_dict:
            continue

        channel = gene_type_dict[gene_type] - 1
        rel_s = max(0, rel_s)
        rel_e = min(seq_len, rel_e)

        if rel_s < rel_e:
            anno[channel, rel_s:rel_e] = 1

    return anno


# =========================================================
# 5) Sequence extraction
# =========================================================
def fetch_sequence(sequence_dict, chrom: str, start: int, end: int, target_len: int) -> np.ndarray:
    chrom = map_chr_to_fasta_key(chrom, sequence_dict)
    if chrom not in sequence_dict:
        return np.array([], dtype=np.int8)

    seq_len = len(sequence_dict[chrom].seq)
    start, end = clamp_window(seq_len, start, end, target_len)

    if end - start != target_len:
        return np.array([], dtype=np.int8)

    seq = str(sequence_dict[chrom].seq[start:end]).upper()
    return one_hot(seq, strict_filter=STRICT_FILTER)


# =========================================================
# 6) Save helpers
# =========================================================
def save_chunk(out_path, chunk_id, e_seqs, p_seqs, labels, e_annos, p_annos):
    enhancer_path = os.path.join(out_path, f"enhancer_Seq_{chunk_id}.npz")
    promoter_path = os.path.join(out_path, f"promoter_Seq_{chunk_id}.npz")

    np.savez(
        enhancer_path,
        label=np.array(labels, dtype=np.int8),
        sequence=np.array(e_seqs, dtype=np.int8),
        annotation=np.array(e_annos, dtype=np.int8),
    )
    np.savez(
        promoter_path,
        label=np.array(labels, dtype=np.int8),
        sequence=np.array(p_seqs, dtype=np.int8),
        annotation=np.array(p_annos, dtype=np.int8),
    )

    print(f"[Saved] {enhancer_path}")
    print(f"[Saved] {promoter_path}")


# =========================================================
# 7) Main processing
# =========================================================
def process_and_save(sequence_dict, gene_features, df, subset_tag, augment_positive=False):
    out_path = os.path.join(BASE_PATH, subset_tag)
    os.makedirs(out_path, exist_ok=True)

    print(f"[Process] subset={subset_tag} rows={len(df)} -> {out_path}")

    e_seqs, p_seqs, labels = [], [], []
    e_annos, p_annos = [], []
    counter, cid = 0, 0

    dropped_invalid_seq = 0
    dropped_bad_chr = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Encoding {subset_tag}"):
        raw_chrom_e = normalize_chr(row["enhancer_chrom"])
        raw_chrom_p = normalize_chr(row["promoter_chrom"])

        chrom_e = map_chr_to_fasta_key(raw_chrom_e, sequence_dict)
        chrom_p = map_chr_to_fasta_key(raw_chrom_p, sequence_dict)

        start_e = int(row["enhancer_start"])
        end_e = int(row["enhancer_end"])
        start_p = int(row["promoter_start"])
        end_p = int(row["promoter_end"])

        label = int(row["label"])

        if chrom_e not in sequence_dict or chrom_p not in sequence_dict:
            dropped_bad_chr += 1
            continue

        repeats = POS_AUG_TIMES if (augment_positive and label == 1) else 1

        for _ in range(repeats):
            if augment_positive and label == 1:
                aug_start_e, aug_end_e = random_resize_window(start_e, end_e, ENHANCER_LEN)
                aug_start_p, aug_end_p = random_resize_window(start_p, end_p, PROMOTER_LEN)
            else:
                aug_start_e, aug_end_e = start_e, end_e
                aug_start_p, aug_end_p = start_p, end_p

            e_hot = fetch_sequence(sequence_dict, raw_chrom_e, aug_start_e, aug_end_e, ENHANCER_LEN)
            p_hot = fetch_sequence(sequence_dict, raw_chrom_p, aug_start_p, aug_end_p, PROMOTER_LEN)

            if len(e_hot) != ENHANCER_LEN * 4:
                dropped_invalid_seq += 1
                continue
            if len(p_hot) != PROMOTER_LEN * 4:
                dropped_invalid_seq += 1
                continue

            e_anno = get_annotation_matrix(gene_features, raw_chrom_e, aug_start_e, aug_end_e, ENHANCER_LEN)
            p_anno = get_annotation_matrix(gene_features, raw_chrom_p, aug_start_p, aug_end_p, PROMOTER_LEN)

            e_seqs.append(e_hot)
            p_seqs.append(p_hot)
            labels.append(label)
            e_annos.append(e_anno)
            p_annos.append(p_anno)

            counter += 1
            if counter >= CHUNK_SIZE:
                save_chunk(out_path, cid, e_seqs, p_seqs, labels, e_annos, p_annos)
                e_seqs, p_seqs, labels, e_annos, p_annos = [], [], [], [], []
                counter = 0
                cid += 1

    if len(labels) > 0:
        save_chunk(out_path, "final", e_seqs, p_seqs, labels, e_annos, p_annos)

    print(f"[Done] subset={subset_tag}")
    print(f"[Stats] dropped_invalid_seq={dropped_invalid_seq}")
    print(f"[Stats] dropped_bad_chr={dropped_bad_chr}")
    print(f"[Stats] kept={len(labels)}")
    if len(labels) > 0:
        print(f"[Stats] label_dist={pd.Series(labels).value_counts().to_dict()}")


# =========================================================
# 8) Main
# =========================================================
def main():
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    print("[INFO] Loading GFF features...")
    gene_features = get_refseq_gff(REF_GFF, include_types)

    print("[INFO] Loading FASTA...")
    with open(REF_FASTA) as f:
        sequence_dict = SeqIO.to_dict(SeqIO.parse(f, "fasta"))

    print("[INFO] Example FASTA keys:", list(sequence_dict.keys())[:5])

    pairs_path = os.path.join(BASE_PATH, "pairs.csv")
    if not os.path.exists(pairs_path):
        raise FileNotFoundError(f"pairs.csv not found: {pairs_path}")

    df_master = pd.read_csv(pairs_path)
    if "label" not in df_master.columns:
        raise ValueError("pairs.csv must contain a 'label' column.")

    print("[INFO] full pairs shape:", df_master.shape)
    print("[INFO] full label dist:", df_master["label"].value_counts().to_dict())

    df_train_pool, df_test = train_test_split(
        df_master,
        test_size=0.1,
        random_state=RANDOM_STATE,
        stratify=df_master["label"],
        shuffle=True,
    )
    df_train_pool = df_train_pool.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    print("[INFO] train_pool shape:", df_train_pool.shape)
    print("[INFO] test_pool shape:", df_test.shape)
    print("[INFO] train_pool label dist:", df_train_pool["label"].value_counts().to_dict())
    print("[INFO] test_pool label dist:", df_test["label"].value_counts().to_dict())

    if TASK == "train":
        df_frac10 = df_train_pool.sample(frac=0.10, random_state=RANDOM_STATE).reset_index(drop=True)
        df_frac20 = df_train_pool.sample(frac=0.20, random_state=RANDOM_STATE).reset_index(drop=True)
        df_frac50 = df_train_pool.sample(frac=0.50, random_state=RANDOM_STATE).reset_index(drop=True)

        process_and_save(sequence_dict, gene_features, df_frac10, "frac10_clean", augment_positive=True)
        process_and_save(sequence_dict, gene_features, df_frac20, "frac20_clean", augment_positive=True)
        process_and_save(sequence_dict, gene_features, df_frac50, "frac50_clean", augment_positive=True)

    elif TASK == "test":
        process_and_save(sequence_dict, gene_features, df_test, "test_clean", augment_positive=False)

    else:
        raise ValueError("TASK must be 'train' or 'test'.")


if __name__ == "__main__":
    main()