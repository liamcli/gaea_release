"""
Evaluation routine for training final architectures from DARTS search space.
Do not use for NASBENCH-201 or NASBENCH-1SHOT1 search spaces.
"""
import os
import sys
import time
import glob
import numpy as np
import random
import torch
import train_utils
import aws_utils
import logging
import argparse

import hydra
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from search_spaces.darts.model import NetworkCIFAR as Network


@hydra.main(config_path="../configs/cnn/config_eval.yaml", strict=False)
def main(args):
    s3_bucket = args.run.s3_bucket
    log = os.path.join(os.getcwd(), "log.txt")

    if s3_bucket is not None:
        aws_utils.download_from_s3(log, s3_bucket, log)

    train_utils.set_up_logging(log)

    CIFAR_CLASSES = 10

    if not torch.cuda.is_available():
        logging.info("no gpu device available")
        sys.exit(1)

    try:
        aws_utils.download_from_s3(
            "cnn_genotypes.txt", s3_bucket, "/tmp/cnn_genotypes.txt"
        )

        with open("/code/nas-theory/cnn/search_spaces/darts/genotypes.py", "a") as f:
            with open("/tmp/cnn_genotypes.txt") as archs:
                f.write("\n")
                for line in archs:
                    if "Genotype" in line:
                        f.write(line)
        print("Downloaded genotypes from aws.")
    except Exception as e:
        print(e)

    # Importing here because need to get the latest genotypes before importing.
    from search_spaces.darts import genotypes

    rng_seed = train_utils.RNGSeed(args.run.seed)

    torch.cuda.set_device(args.run.gpu)
    logging.info("gpu device = %d" % args.run.gpu)
    logging.info("args = %s", args.pretty())

    print(dir(genotypes))

    genotype = eval("genotypes.%s" % args.train.arch)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    model = Network(
        args.train.init_channels,
        CIFAR_CLASSES,
        args.train.layers,
        args.train.auxiliary,
        genotype,
    )
    model = model.cuda()

    optimizer, scheduler = train_utils.setup_optimizer(model, args)

    logging.info("param size = %fMB", train_utils.count_parameters_in_MB(model))
    total_params = sum(x.data.nelement() for x in model.parameters())
    logging.info("Model total parameters: {}".format(total_params))

    try:
        start_epochs, _ = train_utils.load(
            os.getcwd(), rng_seed, model, optimizer, s3_bucket=s3_bucket
        )
        scheduler.last_epoch = start_epochs - 1
    except Exception as e:
        print(e)
        start_epochs = 0

    num_train, num_classes, train_queue, valid_queue = train_utils.create_data_queues(
        args, eval_split=True
    )

    for epoch in range(start_epochs, args.run.epochs):
        logging.info("epoch %d lr %e", epoch, scheduler.get_lr()[0])
        model.drop_path_prob = args.train.drop_path_prob * epoch / args.run.epochs

        train_acc, train_obj = train(args, train_queue, model, criterion, optimizer)
        logging.info("train_acc %f", train_acc)

        valid_acc, valid_obj = train_utils.infer(
            valid_queue, model, criterion, report_freq=args.run.report_freq
        )
        logging.info("valid_acc %f", valid_acc)

        train_utils.save(
            os.getcwd(), epoch + 1, rng_seed, model, optimizer, s3_bucket=s3_bucket
        )
        scheduler.step()


def train(args, train_queue, model, criterion, optimizer):
    objs = train_utils.AvgrageMeter()
    top1 = train_utils.AvgrageMeter()
    top5 = train_utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        input = Variable(input, requires_grad=False).cuda()
        target = Variable(target, requires_grad=False).cuda()

        optimizer.zero_grad()
        logits, logits_aux = model(input)
        loss = criterion(logits, target)
        if args.train.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.train.auxiliary_weight * loss_aux
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.train.grad_clip)
        optimizer.step()

        prec1, prec5 = train_utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.run.report_freq == 0:
            logging.info("train %03d %e %f %f", step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == "__main__":
    main()
