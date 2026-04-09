# -*- coding: utf-8 -*-
"""
00_epi_prepare_with_annotation_frac_colab.py

Preprocessing with explicit --frac tag in output naming, so you can keep your old usage style.

Example
-------
python 00_epi_prepare_with_annotation_frac_colab.py --cell tB --type P-E --frac frac10_clean
"""
import os
import random
import argparse
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO

PROJECT_ROOT = "/content/drive/MyDrive/LOGO_project/test/03_LOGO_EPI"
REF_FASTA = "/content/drive/MyDrive/LOGO_project/test/03_LOGO_EPI/data/hg19/GCF_000001405.25_GRCh37.p13_genomic.fna"
REF_GFF = "/content/drive/MyDrive/LOGO_project/test/03_LOGO_EPI/data/hg19/GCF_000001405.25_GRCh37.p13_genomic.gff"

import sys
sys.path.insert(0, "/content/drive/MyDrive/LOGO_project/test")
from bgi.common.genebank_utils import get_refseq_gff, get_gene_feature_array

ENHANCER_LEN = 2000
PROMOTER_LEN = 1000

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
    "chr1": "NC_000001.10", "chr2": "NC_000002.11", "chr3": "NC_000003.11", "chr4": "NC_000004.11",
    "chr5": "NC_000005.9", "chr6": "NC_000006.11", "chr7": "NC_000007.13", "chr8": "NC_000008.10",
    "chr9": "NC_000009.11", "chr10": "NC_000010.10", "chr11": "NC_000011.9", "chr12": "NC_000012.11",
    "chr13": "NC_000013.10", "chr14": "NC_000014.8", "chr15": "NC_000015.9", "chr16": "NC_000016.9",
    "chr17": "NC_000017.10", "chr18": "NC_000018.9", "chr19": "NC_000019.9", "chr20": "NC_000020.10",
    "chr21": "NC_000021.8", "chr22": "NC_000022.10", "chrX": "NC_000023.10", "chrY": "NC_000024.9",
    "chrM": "NC_012920.1",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cell", default="tB")
    p.add_argument("--type", default="P-E")
    p.add_argument("--frac", default="frac10_clean")
    p.add_argument("--chunk-size", type=int, default=5000)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--strict-filter", action="store_true", default=True)
    p.add_argument("--no-strict-filter", dest="strict_filter", action="store_false")
    p.add_argument("--positive-augment-times", type=int, default=1)
    return p.parse_args()


def normalize_chr(chrom: str) -> str:
    chrom = str(chrom)
    return chrom if chrom.startswith("chr") else f"chr{chrom}"


def map_chr_to_fasta_key(chrom: str, sequence_dict: dict) -> str:
    chrom = normalize_chr(chrom)
    if chrom in sequence_dict:
        return chrom
    if chrom in CHR_TO_NC and CHR_TO_NC[chrom] in sequence_dict:
        return CHR_TO_NC[chrom]
    return chrom


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


def get_annotation_matrix(gene_features, chr_name: str, start: int, end: int, seq_len: int) -> np.ndarray:
    anno = np.zeros((ANNOTATION_SIZE, seq_len), dtype=np.int8)
    feature_list = get_gene_feature_array(gene_features, chr_name, start, end)
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


def fetch_sequence(sequence_dict, chrom: str, start: int, end: int, target_len: int, strict_filter: bool) -> np.ndarray:
    chrom = map_chr_to_fasta_key(chrom, sequence_dict)
    if chrom not in sequence_dict:
        return np.array([], dtype=np.int8)

    seq_len = len(sequence_dict[chrom].seq)
    start, end = clamp_window(seq_len, start, end, target_len)
    if end - start != target_len:
        return np.array([], dtype=np.int8)

    seq = str(sequence_dict[chrom].seq[start:end]).upper()
    return one_hot(seq, strict_filter=strict_filter)


def save_chunk(out_path, frac_tag, chunk_id, e_seqs, p_seqs, labels, e_annos, p_annos):
    enhancer_path = os.path.join(out_path, f"enhancer_raw_{frac_tag}_{chunk_id}.npz")
    promoter_path = os.path.join(out_path, f"promoter_raw_{frac_tag}_{chunk_id}.npz")

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


def build_augmented_df(df: pd.DataFrame, pos_times: int, random_state: int) -> pd.DataFrame:
    if pos_times <= 1:
        return df.copy().reset_index(drop=True)

    rng = np.random.default_rng(random_state)
    rows = []
    for _, row in df.iterrows():
        label = int(row["label"])
        repeat = pos_times if label == 1 else 1
        for _ in range(repeat):
            new_row = row.copy()

            e_start, e_end = int(row["enhancer_start"]), int(row["enhancer_end"])
            p_start, p_end = int(row["promoter_start"]), int(row["promoter_end"])

            e_len = e_end - e_start
            p_len = p_end - p_start

            if e_len < ENHANCER_LEN:
                delta = ENHANCER_LEN - e_len
                shift_left = int(rng.integers(0, delta + 1))
                new_row["enhancer_start"] = e_start - shift_left
                new_row["enhancer_end"] = new_row["enhancer_start"] + ENHANCER_LEN
            else:
                new_row["enhancer_start"] = e_start
                new_row["enhancer_end"] = e_end

            if p_len < PROMOTER_LEN:
                delta = PROMOTER_LEN - p_len
                shift_left = int(rng.integers(0, delta + 1))
                new_row["promoter_start"] = p_start - shift_left
                new_row["promoter_end"] = new_row["promoter_start"] + PROMOTER_LEN
            else:
                new_row["promoter_start"] = p_start
                new_row["promoter_end"] = p_end

            rows.append(new_row)

    out = pd.DataFrame(rows)
    out = out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return out


def process_and_save(sequence_dict, gene_features, df, out_path, frac_tag, chunk_size, strict_filter):
    os.makedirs(out_path, exist_ok=True)
    print(f"[Process] rows={len(df)} frac={frac_tag} -> {out_path}")

    e_seqs, p_seqs, labels = [], [], []
    e_annos, p_annos = [], []
    counter, cid = 0, 0

    dropped_invalid_seq = 0
    dropped_bad_chr = 0
    kept_total = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Encoding"):
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

        e_hot = fetch_sequence(sequence_dict, raw_chrom_e, start_e, end_e, ENHANCER_LEN, strict_filter)
        p_hot = fetch_sequence(sequence_dict, raw_chrom_p, start_p, end_p, PROMOTER_LEN, strict_filter)

        if len(e_hot) != ENHANCER_LEN * 4 or len(p_hot) != PROMOTER_LEN * 4:
            dropped_invalid_seq += 1
            continue

        e_anno = get_annotation_matrix(gene_features, raw_chrom_e, start_e, end_e, ENHANCER_LEN)
        p_anno = get_annotation_matrix(gene_features, raw_chrom_p, start_p, end_p, PROMOTER_LEN)

        e_seqs.append(e_hot)
        p_seqs.append(p_hot)
        labels.append(label)
        e_annos.append(e_anno)
        p_annos.append(p_anno)

        counter += 1
        kept_total += 1
        if counter >= chunk_size:
            save_chunk(out_path, frac_tag, cid, e_seqs, p_seqs, labels, e_annos, p_annos)
            e_seqs, p_seqs, labels, e_annos, p_annos = [], [], [], [], []
            counter = 0
            cid += 1

    if labels:
        save_chunk(out_path, frac_tag, "final", e_seqs, p_seqs, labels, e_annos, p_annos)

    print(f"[Done] kept={kept_total} dropped_invalid_seq={dropped_invalid_seq} dropped_bad_chr={dropped_bad_chr}")


def main():
    args = parse_args()
    random.seed(args.random_state)
    np.random.seed(args.random_state)

    base_path = os.path.join(PROJECT_ROOT, "data", "data", args.cell, args.type)
    pairs_path = os.path.join(base_path, "pairs.csv")
    out_path = os.path.join(base_path, "raw_annotation13_oldcompare_frac")

    if not os.path.exists(pairs_path):
        raise FileNotFoundError(f"pairs.csv not found: {pairs_path}")

    print("[INFO] Loading reference fasta...")
    sequence_dict = SeqIO.to_dict(SeqIO.parse(open(REF_FASTA), "fasta"))
    print("[INFO] Loading GFF features...")
    gene_features = get_refseq_gff(REF_GFF, include_types)

    df = pd.read_csv(pairs_path)
    print(f"[INFO] pairs rows={len(df)}")
    df = build_augmented_df(df, args.positive_augment_times, args.random_state)
    print(f"[INFO] after positive augmentation rows={len(df)}")

    process_and_save(
        sequence_dict=sequence_dict,
        gene_features=gene_features,
        df=df,
        out_path=out_path,
        frac_tag=args.frac,
        chunk_size=args.chunk_size,
        strict_filter=args.strict_filter,
    )

    print(f"[OK] annotation_size={ANNOTATION_SIZE}")
    print(f"[OUT] {out_path}")


if __name__ == "__main__":
    main()
