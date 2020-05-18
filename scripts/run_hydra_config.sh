#!/bin/bash

export PATH=/code/AutoDL/lib/:$PATH
cd /code/nas-theory/cnn
python train_search.py \
	mode=$1 \
        nas_algo=$2 \
	search_config=$3 \
	run.seed=$4 \
	run.epochs=$5 \
    run.dataset=$6 \
    search.single_level=$7 \
    search.exclude_zero=$8 \
    run.s3_bucket=$9 ${10} ${11} ${12} ${13}
