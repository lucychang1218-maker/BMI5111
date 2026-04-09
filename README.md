This repository contains code for reproducing and adapting the LOGO model for two non-coding genome interpretation tasks: 
promoter identification and enhancer-promoter interaction (EPI) prediction, evaluated under resource-constrained conditions.

Promoter Identification
1. Data Preparation

Download GCF_000001405.39_GRCh38.p13_genomic.gff from https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.39_GRCh38.p13/GCF_000001405.39_GRCh38.p13_genomic.gff.gz and unzip to /data/hg38/GCF_000001405.39_GRCh38.p13_genomic.gff
Download EPDnew datasets (BOTH, TATA_BOX, NO_TATA_BOX) from https://epd.epfl.ch/EPD_download.php and generate tfrecord:

python 00_EPDnew_data_prepare.py



2. Training with Knowledge

python 02_PromID_trainer_knowledge.py

3. Training with Freezing Strategy

python 02_PromID_train_knowledge_freeze_transformer_head.py

4. Evaluation Metrics

promoter identification.xls


EPI Prediction
1. Data Download

Download B cells (tB), monocytes (Mon), foetal thymus (FoeT), total CD4+ T cells (tCD4), naive CD4+ T cells (nCD4), and total CD8+ T cells (tCD8) from https://github.com/liwenran/DeepTACT, which derived from https://osf.io/u8tzp/ (Javierre et al. 2016).
From https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.25_GRCh37.p13/ download GCF_000001405.25_GRCh37.p13_genomic.fna.gz and unzip, for example to /data/hg19/GCF_000001405.25_GRCh37.p13_genomic.fna
Download GCF_000001405.39_GRCh38.p13_genomic.gff from https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.39_GRCh38.p13/GCF_000001405.39_GRCh38.p13_genomic.gff.gz and unzip to /data/hg38/GCF_000001405.39_GRCh38.p13_genomic.gff

2. Data Processing and Sequence Generation (~30 GB)

python 00_modified_DataPrepare.py tB P-E
python 00_modified_DataPrepare.py Mon P-E
python 00_modified_DataPrepare.py FoeT P-E
python 00_modified_DataPrepare.py tCD4 P-E
python 00_modified_DataPrepare.py nCD4 P-E
python 00_modified_DataPrepare.py tCD8 P-E

3. Generate k-mer Training Sequences

python 02_modified_DataPrepare_Ngram.py

4. Training and Predicting EPI (Original-like Pipeline)

python 04_LOGO_EPI_train_conv1d_concat_atcg_gene_type.py

5. Training and Predicting EPI (Heavily Modified Pipeline)

python 04_modified_LOGO_EPI_train_conv1d_concat_atcg_gene_type.py
