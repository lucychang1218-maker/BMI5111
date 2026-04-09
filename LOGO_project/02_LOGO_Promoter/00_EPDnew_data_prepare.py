# -*- coding:utf-8 -*-
from pyfaidx import Fasta
import random
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

sys.path.append("../")
from bgi.common.refseq_utils import get_word_dict_for_n_gram_alphabet
from bgi.common.genebank_utils import get_refseq_gff, get_gene_feature_array


# =========================
# 0) Config
# =========================
include_types = [
    'enhancer',
    # 'promoter',
    'pseudogene',
    'insulator',
    'conserved_region',
    'protein_binding_site',
    'DNAseI_hypersensitive_site',
    'nucleotide_cleavage_site',
    'silencer',
    'gene',
    'exon',
    'CDS',
]

chr_dict_hg38 = {
    "NC_000001.11": "chr1",
    "NC_000002.12": "chr2",
    "NC_000003.12": "chr3",
    "NC_000004.12": "chr4",
    "NC_000005.10": "chr5",
    "NC_000006.12": "chr6",
    "NC_000007.14": "chr7",
    "NC_000008.11": "chr8",
    "NC_000009.12": "chr9",
    "NC_000010.11": "chr10",
    "NC_000011.10": "chr11",
    "NC_000012.12": "chr12",
    "NC_000013.11": "chr13",
    "NC_000014.9": "chr14",
    "NC_000015.10": "chr15",
    "NC_000016.10": "chr16",
    "NC_000017.11": "chr17",
    "NC_000018.10": "chr18",
    "NC_000019.10": "chr19",
    "NC_000020.11": "chr20",
    "NC_000021.9": "chr21",
    "NC_000022.11": "chr22",
    "NC_000023.11": "chrX",
    "NC_000024.10": "chrY",
}


# =========================
# 1) Utils
# =========================
def sample_fraction(sequences, annotations, labels, frac=0.05, seed=42):
    if frac >= 1.0:
        return sequences, annotations, labels
    n = len(labels)
    if n == 0:
        return [], [], []
    rng = np.random.default_rng(seed)
    k = max(1, int(n * frac))
    idx = rng.choice(n, size=k, replace=False)
    return (
        [sequences[i] for i in idx],
        [annotations[i] for i in idx],
        [labels[i] for i in idx],
    )


def get_negative(sequence: str,
                 TSS_position=5000,
                 low_position=200,
                 high_position=400,
                 ngram=3,
                 max_tries=2000):
    seq_len = len(sequence)
    forbidden_left = TSS_position - low_position
    forbidden_right = TSS_position + high_position

    for _ in range(max_tries):
        center = random.randint(0, seq_len - 1)
        if forbidden_left <= center <= forbidden_right:
            continue
        start = center - low_position
        end = center + high_position + (ngram - 1)
        if start < 0 or end >= seq_len:
            continue
        return sequence[start:end + 1], center

    return "", -1


def _is_valid_dna(seq: str, skip_n: bool = True):
    if not seq:
        return False
    s = seq.upper()
    if skip_n and 'N' in s:
        return False
    return set(s).issubset(set("ATCG"))


def _frac_tag(frac: float) -> str:
    return f"frac{int(round(frac * 100))}"


# =========================
# 2) Core: to NPZ
# =========================
def process_raw_text_with_annotation(
    sequences,
    annotations,
    labels,
    seq_size=600,
    ngram=3,
    stride=1,
    filter_txt=None,
    skip_n: bool = True,
    word_dict: dict = None,
    output_path: str = './',
    task_name: str = 'train',
    gene_type_dict: dict = None,
    frac: float = 1.0,
    seed: int = 42,
):
    if word_dict is None:
        raise ValueError("word_dict cannot be None.")
    if gene_type_dict is None:
        raise ValueError("gene_type_dict cannot be None.")

    slice_seq_data = []
    slice_anno_data = []
    slice_label_data = []
    dropped_invalid = 0

    for seq, anno, label in zip(sequences, annotations, labels):
        seq = str(seq).upper()

        if filter_txt is not None and seq.startswith(filter_txt):
            continue

        if not _is_valid_dna(seq, skip_n=skip_n):
            dropped_invalid += 1
            continue

        if len(seq) < seq_size + (ngram - 1):
            dropped_invalid += 1
            continue

        seq_number = []
        for jj in range(0, seq_size, stride):
            kmer = seq[jj:jj + ngram]
            if len(kmer) == ngram:
                seq_number.append(word_dict.get(kmer, 0))
        slice_seq_data.append(seq_number)
        slice_label_data.append(int(label))

        anno_position = np.zeros((len(gene_type_dict.keys()) + 2, seq_size), dtype=np.int8)
        if anno:
            for gene_type, start, end in anno:
                gene_idx = gene_type_dict.get(gene_type, 0)
                s = int(start)
                e = int(end)
                if e > 0 and s < seq_size:
                    anno_position[gene_idx, max(0, s):min(seq_size, e)] = 1
        slice_anno_data.append(anno_position)

    os.makedirs(output_path, exist_ok=True)
    frac_tag = _frac_tag(frac)
    save_path = os.path.join(output_path, f'{task_name}_{ngram}_gram_{frac_tag}.npz')

    if slice_seq_data and slice_label_data:
        np.savez_compressed(
            save_path,
            sequence=slice_seq_data,
            annotation=slice_anno_data,
            label=slice_label_data,
            meta=np.array([{
                "task_name": task_name,
                "ngram": ngram,
                "seq_size": seq_size,
                "stride": stride,
                "skip_n": skip_n,
                "frac": frac,
                "seed": seed,
                "kept": int(len(slice_label_data)),
                "dropped_invalid": int(dropped_invalid),
                "label_dist": dict(Counter(slice_label_data)),
            }], dtype=object)
        )
        print(f"[Saved] {save_path}")
        print(f"[Info] kept={len(slice_label_data)} dropped_invalid={dropped_invalid} labels={Counter(slice_label_data)}")
    else:
        print(f"[Warn] No data saved for task={task_name}. kept=0 dropped_invalid={dropped_invalid}")


# =========================
# 3) Load EPDnew + annotation
# =========================
def get_epdnew_data_with_annotation(
    fasta_file: str,
    bed_file: str,
    gff_file: str,
    TSS_position=5000,
    low_position=200,
    high_position=400,
    ngram=3,
    skip_n: bool = True,
    frac: float = 1.0,
    seed: int = 42,
    verbose_checks: bool = True
):
    if not (os.path.exists(fasta_file) and os.path.exists(bed_file) and os.path.exists(gff_file)):
        print(f"[Warn] missing files:\n  fasta={fasta_file}\n  bed={bed_file}\n  gff={gff_file}")
        return [], [], [], 0

    promoter_df = pd.read_csv(
        bed_file,
        sep='\t',
        header=None,
        names=['Ref', 'TSS', 'Location', 'V', 'Name', 'Strand'],
    )

    epdnew = Fasta(fasta_file)
    epd_ids = [str(k) for k in epdnew.keys()]

    promoter_df = promoter_df.iloc[:len(epd_ids)].copy()
    promoter_df['epd_id'] = epd_ids

    promoter_df.drop(columns=['V'], inplace=True, errors='ignore')
    promoter_df.sort_values(by=['Ref', 'Location'], ascending=[True, True], inplace=True)

    chr_gff_dict = get_refseq_gff(gff_file, include_types)
    chr_convert_dict = {v: k for k, v in chr_dict_hg38.items()}

    sequences, annotations, labels = [], [], []
    window_size = low_position + high_position

    kept_pos = kept_neg = 0
    dropped_pos = dropped_neg = 0

    for _, row in tqdm(promoter_df.iterrows(), total=len(promoter_df), desc=f"Promoters (k={ngram})"):
        epd_id = row['epd_id']
        ref = row['Ref']
        location = int(row['Location'])

        convert_ref = chr_convert_dict.get(ref, '')
        sequence = epdnew[epd_id]

        # positive
        start = TSS_position - low_position
        end = TSS_position + high_position + (ngram - 1)
        positive_seq = str(sequence[start:end]).upper()

        ann_start = location - low_position
        ann_end = location + high_position
        pos_anno = get_gene_feature_array(chr_gff_dict, convert_ref, ann_start, ann_end) or []

        if _is_valid_dna(positive_seq, skip_n=skip_n) and len(positive_seq) >= window_size + (ngram - 1):
            sequences.append(positive_seq)
            annotations.append(pos_anno)
            labels.append(1)
            kept_pos += 1
        else:
            dropped_pos += 1

        # negative
        negative_seq, neg_center = get_negative(
            sequence=str(sequence).upper(),
            TSS_position=TSS_position,
            low_position=low_position,
            high_position=high_position,
            ngram=ngram,
            max_tries=2000
        )
        if negative_seq and neg_center != -1:
            shift = neg_center - TSS_position
            ann_start2 = (location - low_position) + shift
            ann_end2 = (location + high_position) + shift
            neg_anno = get_gene_feature_array(chr_gff_dict, convert_ref, ann_start2, ann_end2) or []

            if _is_valid_dna(negative_seq, skip_n=skip_n) and len(negative_seq) >= window_size + (ngram - 1):
                sequences.append(negative_seq)
                annotations.append(neg_anno)
                labels.append(0)
                kept_neg += 1
            else:
                dropped_neg += 1
        else:
            dropped_neg += 1

    if verbose_checks:
        print(f"[Check] pos kept/dropped = {kept_pos}/{dropped_pos}")
        print(f"[Check] neg kept/dropped = {kept_neg}/{dropped_neg}")
        print(f"[Check] total kept = {len(labels)}  label_dist = {Counter(labels)}")

    sequences, annotations, labels = sample_fraction(sequences, annotations, labels, frac=frac, seed=seed)

    if verbose_checks:
        print(f"[Sample] frac={frac} -> kept={len(labels)} label_dist={Counter(labels)} (seed={seed})")

    return sequences, annotations, labels, window_size


# =========================
# 4) Main: run 3 datasets * multiple fracs
# =========================
if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)

    gff_file = './data/EPDnew/GCF_000001405.39_GRCh38.p13_genomic.gff'

    ngram = 5
    stride = 1
    word_dict = get_word_dict_for_n_gram_alphabet(n_gram=ngram)

    # ✅ 你截图里实际存在的文件（按你之前的映射先写死）
    DATASETS = [
        {
            "task_name": "epdnew_BOTH_Knowledge",
            "fasta": "./data/EPDnew/hg38_QMNCo.fa",
            "bed":   "./data/EPDnew/human_epdnew_FzZ6q.bed",
        },
        {
            "task_name": "epdnew_NO_TATA_BOX_Knowledge",
            "fasta": "./data/EPDnew/hg38_oBvPw.fa",
            "bed":   "./data/EPDnew/human_epdnew_4hlzk.bed",
        },
        {
            "task_name": "epdnew_TATA_BOX_Knowledge",
            "fasta": "./data/EPDnew/hg38_Q1zFL.fa",
            "bed":   "./data/EPDnew/human_epdnew_CAK3S.bed",
        },
    ]

    # gene_type_dict
    gene_type_dict = {}
    idx = 1
    for gene_type in include_types:
        gene_type_dict[gene_type] = idx
        idx += 1

    out_dir = f'./data/{ngram}_gram_11_knowledge'
    os.makedirs(out_dir, exist_ok=True)

    FRACS = [0.05, 0.10]
    SKIP_N = True
    SEED = 42

    for ds in DATASETS:
        task_name = ds["task_name"]
        fasta_file = ds["fasta"]
        bed_file = ds["bed"]

        # 先检查文件是否存在（不存在就跳过，不会 misleading）
        if not os.path.exists(fasta_file):
            print(f"[Skip] missing fasta: {fasta_file}")
            continue
        if not os.path.exists(bed_file):
            print(f"[Skip] missing bed: {bed_file}")
            continue
        if not os.path.exists(gff_file):
            print(f"[Skip] missing gff: {gff_file}")
            break

        for FRAC in FRACS:
            print("\n" + "=" * 80)
            print(f"[RUN] task={task_name} ngram={ngram} frac={FRAC} seed={SEED} skip_n={SKIP_N}")
            print("=" * 80)

            sequences, annotations, labels, window_size = get_epdnew_data_with_annotation(
                fasta_file=fasta_file,
                bed_file=bed_file,
                gff_file=gff_file,
                ngram=ngram,
                skip_n=SKIP_N,
                frac=FRAC,
                seed=SEED,
                verbose_checks=True
            )

            process_raw_text_with_annotation(
                sequences=sequences,
                annotations=annotations,
                labels=labels,
                seq_size=window_size,   # 600
                ngram=ngram,
                stride=stride,
                skip_n=SKIP_N,
                word_dict=word_dict,
                output_path=out_dir,
                task_name=task_name,
                gene_type_dict=gene_type_dict,
                frac=FRAC,
                seed=SEED,
            )
