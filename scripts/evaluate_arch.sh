#!/bin/bash
cd /code/nas-theory/cnn
python train_aws.py train.arch=$1 run.seed=$2 train.drop_path_prob=$3 train.batch_size=$4 run.s3_bucket=$5
