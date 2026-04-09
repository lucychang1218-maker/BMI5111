# -*- coding: utf-8 -*-
"""
01_epi_make_kmer_annotation13_frac_colab.py

Keep old preferred usage with --frac tag.
"""
import os
import argparse
import numpy as np

import sys
sys.path.insert(0, "/content/drive/MyDrive/LOGO_project/test")
from bgi.common.refseq_utils import get_word_dict_for_n_gram_number

PROJECT_ROOT = "/content/drive/MyDrive/LOGO_project/test/03_LOGO_EPI"
NUM_SEQ = 4


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cell", default="tB")
    p.add_argument("--type", default="P-E")
    p.add_argument("--ngram", type=int, default=5)
    p.add_argument("--frac", default="frac10_clean")
    return p.parse_args()


def preprocess_onehot_to_kmer(seq_data: np.ndarray, ngram: int, stride: int = 1) -> np.ndarray:
    actg_value = np.array([1, 2, 3, 4])
    n_gram_value = np.ones(ngram)
    for ii in range(ngram):
        n_gram_value[ii] = int(n_gram_value[ii] * (10 ** (ngram - ii - 1)))

    num_word_dict = get_word_dict_for_n_gram_number(n_gram=ngram)

    actg = np.matmul(seq_data, actg_value)
    gene = []
    seq_max_len = len(actg) // ngram // ngram * ngram * ngram
    for kk in range(0, seq_max_len, stride):
        if kk + ngram <= seq_max_len:
            gene.append(actg[kk:kk + ngram])
    gene = np.array(gene)
    gene = np.matmul(gene, n_gram_value)

    gene_seq = []
    for token in gene:
        gene_seq.append(num_word_dict.get(token, 0))
    return np.array(gene_seq, dtype=np.int32)


def load_side(raw_dir: str, prefix: str, frac: str, ngram: int):
    files = sorted([
        os.path.join(raw_dir, f)
        for f in os.listdir(raw_dir)
        if f.startswith(f"{prefix}{frac}_") and f.endswith(".npz")
    ])
    if not files:
        raise FileNotFoundError(f"No files found in {raw_dir} with prefix {prefix}{frac}_")

    x_all, a_all, y_all = [], [], []

    for fn in files:
        loaded = np.load(fn, allow_pickle=True)
        x = loaded["sequence"]
        a = loaded["annotation"]
        y = loaded["label"]

        print(f"[Load] {os.path.basename(fn)} x={x.shape} a={a.shape} y={y.shape}")

        x = np.reshape(x, (x.shape[0], x.shape[1] // NUM_SEQ, NUM_SEQ))

        x_kmer = []
        for i in range(len(x)):
            x_kmer.append(preprocess_onehot_to_kmer(x[i], ngram=ngram, stride=1))
            if i > 0 and i % 10000 == 0:
                print(f"[{prefix}{frac}] processed {i}")

        x_kmer = np.array(x_kmer, dtype=np.int32)

        x_all.append(x_kmer)
        a_all.append(a.astype(np.int8))
        y_all.append(y.astype(np.int8))

    X = np.vstack(x_all)
    A = np.vstack(a_all)
    Y = np.hstack(y_all)
    return X, A, Y


def main():
    args = parse_args()
    base_path = os.path.join(PROJECT_ROOT, "data", "data", args.cell, args.type)
    raw_dir = os.path.join(base_path, "raw_annotation13_oldcompare_frac")
    out_dir = os.path.join(base_path, f"{args.ngram}_gram_oldcompare_13anno_frac")
    os.makedirs(out_dir, exist_ok=True)

    X_e, A_e, Y_e = load_side(raw_dir, "enhancer_raw_", args.frac, args.ngram)
    X_p, A_p, Y_p = load_side(raw_dir, "promoter_raw_", args.frac, args.ngram)

    if len(Y_e) != len(Y_p):
        raise ValueError(f"label length mismatch: enhancer={len(Y_e)} promoter={len(Y_p)}")

    e_out = os.path.join(out_dir, f"enhancer_Seq_{args.ngram}_gram_knowledge_{args.frac}.npz")
    p_out = os.path.join(out_dir, f"promoter_Seq_{args.ngram}_gram_knowledge_{args.frac}.npz")

    np.savez(e_out, sequence=X_e, annotation=A_e, label=Y_e)
    np.savez(p_out, sequence=X_p, annotation=A_p, label=Y_p)

    print("[Saved]", e_out)
    print("[Saved]", p_out)
    print("[Shapes] enhancer", X_e.shape, A_e.shape, Y_e.shape)
    print("[Shapes] promoter", X_p.shape, A_p.shape, Y_p.shape)


if __name__ == "__main__":
    main()
