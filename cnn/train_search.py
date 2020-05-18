import os
import sys
import time
import glob
import numpy as np
import torch
import train_utils
import aws_utils
import random
import copy
import logging
import hydra
import pickle
import torch.nn as nn
import torch.utils
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter


@hydra.main(config_path="../configs/cnn/config.yaml", strict=False)
def main(args):
    np.set_printoptions(precision=3)
    save_dir = os.getcwd()

    log = os.path.join(save_dir, "log.txt")

    # Setup SummaryWriter
    summary_dir = os.path.join(save_dir, "summary")
    if not os.path.exists(summary_dir):
        os.mkdir(summary_dir)
    writer = SummaryWriter(summary_dir)

    if args.run.s3_bucket is not None:
        aws_utils.download_from_s3(log, args.run.s3_bucket, log)

        train_utils.copy_code_to_experiment_dir("/code/nas-theory/cnn", save_dir)
        aws_utils.upload_directory(
            os.path.join(save_dir, "scripts"), args.run.s3_bucket
        )

    train_utils.set_up_logging(log)

    if not torch.cuda.is_available():
        logging.info("no gpu device available")
        sys.exit(1)

    torch.cuda.set_device(args.run.gpu)
    logging.info("gpu device = %d" % args.run.gpu)
    logging.info("args = %s", args.pretty())

    rng_seed = train_utils.RNGSeed(args.run.seed)

    if args.search.method in ["edarts", "gdarts", "eedarts"]:
        if args.search.fix_alphas:
            from architect.architect_edarts_edge_only import (
                ArchitectEDARTS as Architect,
            )
        else:
            from architect.architect_edarts import ArchitectEDARTS as Architect
    elif args.search.method in ["darts", "fdarts"]:
        from architect.architect_darts import ArchitectDARTS as Architect
    elif args.search.method == "egdas":
        from architect.architect_egdas import ArchitectEGDAS as Architect
    else:
        raise NotImplementedError

    if args.search.search_space in ["darts", "darts_small"]:
        from search_spaces.darts.model_search import DARTSNetwork as Network
    elif "nas-bench-201" in args.search.search_space:
        from search_spaces.nasbench_201.model_search import (
            NASBENCH201Network as Network,
        )
    elif args.search.search_space == "pcdarts":
        from search_spaces.pc_darts.model_search import PCDARTSNetwork as Network
    else:
        raise NotImplementedError

    if args.train.smooth_cross_entropy:
        criterion = train_utils.cross_entropy_with_label_smoothing
    else:
        criterion = nn.CrossEntropyLoss()

    num_train, num_classes, train_queue, valid_queue = train_utils.create_data_queues(
        args
    )

    print("dataset: {}, num_classes: {}".format(args.run.dataset, num_classes))

    model = Network(
        args.train.init_channels,
        num_classes,
        args.search.nodes,
        args.train.layers,
        criterion,
        **{
            "auxiliary": args.train.auxiliary,
            "search_space_name": args.search.search_space,
            "exclude_zero": args.search.exclude_zero,
            "track_running_stats": args.search.track_running_stats,
        }
    )
    model = model.cuda()
    logging.info("param size = %fMB", train_utils.count_parameters_in_MB(model))

    optimizer, scheduler = train_utils.setup_optimizer(model, args)

    # TODO: separate args by model, architect, etc
    # TODO: look into using hydra for config files
    architect = Architect(model, args, writer)

    # Try to load a previous checkpoint
    try:
        start_epochs, history = train_utils.load(
            save_dir, rng_seed, model, optimizer, architect, args.run.s3_bucket
        )
        scheduler.last_epoch = start_epochs - 1
        num_train, num_classes, train_queue, valid_queue = train_utils.create_data_queues(
            args
        )
    except Exception as e:
        logging.info(e)
        start_epochs = 0


    best_valid = 0
    for epoch in range(start_epochs, args.run.epochs):
        lr = scheduler.get_lr()[0]
        logging.info("epoch %d lr %e", epoch, lr)

        model.drop_path_prob = args.train.drop_path_prob * epoch / args.run.epochs

        # training
        train_acc, train_obj = train(
            args, train_queue, valid_queue, model, architect, criterion, optimizer, lr,
        )
        architect.baseline = train_obj
        architect.update_history()
        architect.log_vars(epoch, writer)

        if "update_lr_state" in dir(scheduler):
            scheduler.update_lr_state(train_obj)

        logging.info("train_acc %f", train_acc)

        # History tracking
        for vs in [("alphas", architect.alphas), ("edges", architect.edges)]:
            for ct in vs[1]:
                v = vs[1][ct]
                logging.info("{}-{}".format(vs[0], ct))
                logging.info(v)
        # Calling genotypes sets alphas to best arch for EGDAS and MEGDAS
        # so calling here before infer.
        genotype = architect.genotype()
        logging.info("genotype = %s", genotype)

        if not args.search.single_level:
            valid_acc, valid_obj = train_utils.infer(
                valid_queue,
                model,
                criterion,
                report_freq=args.run.report_freq,
                discrete=args.search.discrete,
            )
            if valid_acc > best_valid:
                best_valid = valid_acc
                best_genotype = architect.genotype()
            logging.info("valid_acc %f", valid_acc)

        train_utils.save(
            save_dir,
            epoch + 1,
            rng_seed,
            model,
            optimizer,
            architect,
            save_history=True,
            s3_bucket=args.run.s3_bucket,
        )

        scheduler.step()

    valid_acc, valid_obj = train_utils.infer(
        valid_queue,
        model,
        criterion,
        report_freq=args.run.report_freq,
        discrete=args.search.discrete,
    )
    if valid_acc > best_valid:
        best_valid = valid_acc
        best_genotype = architect.genotype()
    logging.info("valid_acc %f", valid_acc)

    if args.run.s3_bucket is not None:
        filename = "cnn_genotypes.txt"
        aws_utils.download_from_s3(filename, args.run.s3_bucket, filename)

        with open(filename, "a+") as f:
            f.write("\n")
            f.write(
                "{}{}{}{} = {}".format(
                    args.search.search_space,
                    args.search.method,
                    args.run.dataset.replace("-", ""),
                    args.run.seed,
                    best_genotype,
                )
            )
        aws_utils.upload_to_s3(filename, args.run.s3_bucket, filename)
        aws_utils.upload_to_s3(log, args.run.s3_bucket, log)


def train(
    args,
    train_queue,
    valid_queue,
    model,
    architect,
    criterion,
    optimizer,
    lr,
    random_arch=False,
):
    objs = train_utils.AvgrageMeter()
    top1 = train_utils.AvgrageMeter()
    top5 = train_utils.AvgrageMeter()

    for step, datapoint in enumerate(train_queue):
        # The search dataqueue for nas-bench-201  returns both train and valid data
        # when looping through queue.  This is disabled with single level is indicated.
        if "nas-bench-201" in args.search.search_space and not (
            args.search.single_level
        ):
            input, target, input_search, target_search = datapoint
        else:
            input, target = datapoint
            input_search, target_search = next(iter(valid_queue))

        n = input.size(0)

        input = Variable(input, requires_grad=False).cuda()
        target = Variable(target, requires_grad=False).cuda()

        # get a random minibatch from the search queue with replacement
        input_search = Variable(input_search, requires_grad=False).cuda()
        target_search = Variable(target_search, requires_grad=False).cuda()

        model.train()

        # TODO: move architecture args into a separate dictionary within args
        if not random_arch:
            architect.step(
                input,
                target,
                input_search,
                target_search,
                **{
                    "eta": lr,
                    "network_optimizer": optimizer,
                    "unrolled": args.search.unrolled,
                    "update_weights": True,
                }
            )
        # if random_arch or model.architect_type == "snas":
        #    architect.sample_arch_configure_model()

        optimizer.zero_grad()
        architect.zero_arch_var_grad()
        architect.set_model_alphas()
        architect.set_model_edge_weights()

        logits, logits_aux = model(input, discrete=args.search.discrete)
        loss = criterion(logits, target)
        if args.train.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.train.auxiliary_weight * loss_aux

        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.train.grad_clip)
        optimizer.step()

        prec1, prec5 = train_utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.run.report_freq == 0:
            logging.info("train %03d %e %f %f", step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == "__main__":
    main()
