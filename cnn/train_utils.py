import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import os
import sys
from distutils.dir_util import copy_tree
import aws_utils
import pickle
from lr_schedulers import *
from copy import deepcopy

# from genotypes import PRIMITIVES
import logging
import torchvision.datasets as dset
import torchvision.datasets as dset
from torch.autograd import Variable
from collections import deque
import autoaugment.augmentation_transforms as augmentation_transforms
import autoaugment.policies as found_policies
import torchvision.transforms as transforms

# Import AutoDL Nasbench-201 functions for data loading
from lib.datasets.get_dataset_with_transform import get_datasets, get_nas_search_loaders


class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def AutoAugment(img):
    good_policies = found_policies.good_policies()
    policy = good_policies[np.random.choice(len(good_policies))]
    final_img = augmentation_transforms.apply_policy(policy, img)
    return final_img


def _data_transforms_cifar10(args):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    if args.train.cutout:
        train_transform.transforms.append(Cutout(args.train.cutout_length))
    if args.train.autoaugment:
        train_transform.transforms.insert(0, AutoAugment)
    print(train_transform.transforms)

    valid_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD),]
    )
    return train_transform, valid_transform


def count_parameters_in_MB(model):
    return (
        np.sum(
            np.prod(v.size())
            for name, v in model.named_parameters()
            if "auxiliary" not in name
        )
        / 1e6
    )


def drop_path(x, drop_prob):
    if drop_prob > 0.0:
        keep_prob = 1.0 - drop_prob
        mask = Variable(
            torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        )
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def set_up_logging(path):
    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format=log_format,
        datefmt="%m/%d %I:%M:%S %p",
    )
    fh = logging.FileHandler(path)
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)


def label_smoothing(pred, target, eta=0.1):
    """
    Code from https://github.com/mit-han-lab/ProxylessNAS/blob/master/training/utils/bags_of_tricks.py
    See https://arxiv.org/pdf/1512.00567.pdf for more info about label smoothing softmax.

    Args:
      pred: predictions
      target: ints representing class label
      eta: smoothing parameter, eta is spread to other classes
    Return:
      same shape as predictions
    """
    n_classes = pred.size(1)
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros_like(pred)
    onehot_target.scatter_(1, target, 1)
    return onehot_target * (1 - eta) + eta / n_classes * 1


def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(-target * F.log_softmax(pred, dim=-1), 1))


def cross_entropy_with_label_smoothing(pred, target, eta=0.1):
    onehot_target = label_smoothing(pred, target, eta)
    return cross_entropy_for_onehot(pred, onehot_target)


def setup_optimizer(model, args):
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.train.learning_rate,
        momentum=args.train.momentum,
        weight_decay=args.train.weight_decay,
    )

    min_lr = args.train.learning_rate_min
    if args.train.scheduler == "triangle":
        scheduler = TriangleScheduler(
            optimizer,
            0.003,
            0.1,
            min_lr,
            5,
            args.run.epochs,
            backoff_scheduler=scheduler,
        )
    else:
        lr_anneal_cycles = 1
        if "lr_anneal_cycles" in args.train:
            lr_anneal_cycles = args.train.lr_anneal_cycles
        if args.train.scheduler == "cosine":
            scheduler = CosinePowerAnnealing(
                optimizer, 1, lr_anneal_cycles, min_lr, args.run.scheduler_epochs
            )
        elif args.train.scheduler == "powercosine":
            scheduler = CosinePowerAnnealing(
                optimizer, 2, lr_anneal_cycles, min_lr, args.run.scheduler_epochs
            )
        else:
            raise NotImplementedError(
                "lr scheduler not implemented, please select one from [cosine, powercosine, triangle]"
            )

    return optimizer, scheduler


def create_nasbench_201_data_queues(args, eval_split=False):
    assert args.run.dataset in ["cifar10", "cifar100", "ImageNet16-120"]
    path_mapping = {
        "cifar10": "cifar-10-batches-py",
        "cifar100": "cifar-100-python",
        "ImageNet16-120": "ImageNet16",
    }
    train_data, valid_data, xshape, class_num = get_datasets(
        args.run.dataset,
        os.path.join(args.run.data, path_mapping[args.run.dataset]),
        -1,
    )
    if eval_split:
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.train.batch_size,
            shuffle=True,
            num_workers=args.run.n_threads_data,
            pin_memory=True,
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_data,
            batch_size=args.train.batch_size,
            shuffle=False,
            num_workers=args.run.n_threads_data,
            pin_memory=True,
        )
        return len(train_data), class_num, train_loader, valid_loader

    if args.search.single_level:
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.train.batch_size,
            shuffle=True,
            num_workers=args.run.n_threads_data,
            pin_memory=True,
        )
        valid_data = deepcopy(train_data)
        valid_loader = torch.utils.data.DataLoader(
            valid_data,
            batch_size=args.train.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=args.run.n_threads_data,
        )
        num_train = len(train_loader.dataset)
        return num_train, class_num, train_loader, valid_loader

    search_loader, _, valid_loader = get_nas_search_loaders(
        train_data,
        valid_data,
        args.run.dataset,
        os.path.join(args.run.autodl, "configs/nas-benchmark/"),
        args.train.batch_size,
        args.run.n_threads_data,
    )
    num_train = len(search_loader.dataset)
    return num_train, class_num, search_loader, valid_loader


def create_data_queues(args, eval_split=False):
    if "nas-bench-201" in args.search.search_space:
        return create_nasbench_201_data_queues(args, eval_split)

    train_transform, valid_transform = _data_transforms_cifar10(args)
    train_data = dset.CIFAR10(
        root=args.run.data, train=True, download=True, transform=train_transform
    )
    valid_data = dset.CIFAR10(
        root=args.run.data, train=False, download=True, transform=valid_transform
    )
    num_train = 64 * 10 if args.run.test_code else len(train_data)

    if eval_split:
        # This is used if evaluating a single architecture from scratch.
        train_queue = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.train.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=args.run.n_threads_data,
        )

        valid_queue = torch.utils.data.DataLoader(
            valid_data,
            batch_size=args.train.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=args.run.n_threads_data,
        )
    else:
        # This is used during architecture search.
        indices = list(range(num_train))

        if args.search.single_level:
            train_end = num_train
            val_start = 0
        else:
            split = int(np.floor(args.search.train_portion * num_train))
            train_end = split
            val_start = split

        valid_data = deepcopy(train_data)

        train_queue = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.train.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[0:train_end]),
            pin_memory=True,
            num_workers=0,
        )

        valid_queue = torch.utils.data.DataLoader(
            valid_data,
            batch_size=args.train.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(
                np.random.permutation(indices[val_start:])
            ),
            pin_memory=True,
            num_workers=0,
        )
    return num_train, 10, train_queue, valid_queue


class RNGSeed:
    def __init__(self, seed):
        self.seed = seed
        self.set_random_seeds()

    def set_random_seeds(self):
        seed = self.seed
        random.seed(seed)
        np.random.seed(seed)
        cudnn.benchmark = False
        torch.manual_seed(seed)
        cudnn.enabled = True
        cudnn.deterministic = True
        torch.cuda.manual_seed(seed)

    def get_save_states(self):
        rng_states = {
            "random_state": random.getstate(),
            "np_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "torch_cuda_random_state": torch.cuda.get_rng_state_all(),
        }
        return rng_states

    def load_states(self, rng_states):
        random.setstate(rng_states["random_state"])
        np.random.set_state(rng_states["np_random_state"])
        torch.set_rng_state(rng_states["torch_random_state"])
        torch.cuda.set_rng_state_all(rng_states["torch_cuda_random_state"])


def save(
    folder,
    epochs,
    rng_seed,
    model,
    optimizer,
    architect=None,
    save_history=False,
    s3_bucket=None,
):
    """
    Create checkpoint and save to directory.
    TODO: should remove s3 handling and have separate class for that.

    Args:
        folder: save directory
        epochs: number of epochs completed
        rng_state: rng states for np, torch, random, etc.
        model: model object with its own get_save_states method
        optimizer: model optimizer
        architect: architect object with its own get_save_states method
        s3_bucket: s3 bucket name for saving to AWS s3
    """

    checkpoint = {
        "epochs": epochs,
        "rng_seed": rng_seed.get_save_states(),
        "optimizer": optimizer.state_dict(),
        "model": model.get_save_states(),
    }

    if architect is not None:
        checkpoint["architect"] = architect.get_save_states()

    ckpt = os.path.join(folder, "model.ckpt")
    torch.save(checkpoint, ckpt)

    history = None
    if save_history:
        history_file = os.path.join(folder, "history.pkl")
        history = architect.get_history()
        with open(history_file, "wb") as f:
            pickle.dump(history, f)

    log = os.path.join(folder, "log.txt")

    if s3_bucket is not None:
        aws_utils.upload_to_s3(ckpt, s3_bucket, ckpt)
        aws_utils.upload_to_s3(log, s3_bucket, log)
        if history is not None:
            aws_utils.upload_to_s3(history_file, s3_bucket, history_file)
        # try:
        #    summary_dir = os.path.join(folder, 'summary')
        #    for root, dirs, files in os.walk(summary_dir):
        #        for f in files:
        #            path = os.path.join(root, f)
        #            aws_utils.upload_to_s3(path, s3_bucket, path)


def load(folder, rng_seed, model, optimizer, architect=None, s3_bucket=None):
    # Try to download log and ckpt from s3 first to see if a ckpt exists.
    ckpt = os.path.join(folder, "model.ckpt")
    history_file = os.path.join(folder, "history.pkl")
    history = {}

    if s3_bucket is not None:
        aws_utils.download_from_s3(ckpt, s3_bucket, ckpt)
        try:
            aws_utils.download_from_s3(history_file, s3_bucket, history_file)
        except:
            logging.info("history.pkl not in s3 bucket")

    if os.path.exists(history_file):
        with open(history_file, "rb") as f:
            history = pickle.load(f)
        # TODO: update architecture history
        architect.load_history(history)

    checkpoint = torch.load(ckpt)

    epochs = checkpoint["epochs"]
    rng_seed.load_states(checkpoint["rng_seed"])
    model.load_states(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if architect is not None:
        architect.load_states(checkpoint["architect"])

    logging.info("Resumed model trained for %d epochs" % epochs)

    return epochs, history


def infer(valid_queue, model, criterion, report_freq=50, discrete=False):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    model.eval()

    # if arch is not None:
    #    alphas_normal, alphas_reduce = get_weights_from_arch(model, arch)
    #    normal_orig = torch.tensor(model.alphas_normal.data, requires_grad=False)
    #    reduce_orig = torch.tensor(model.alphas_reduce.data, requires_grad=False)
    #    model.alphas_normal.data = alphas_normal
    #    model.alphas_reduce.data = alphas_reduce
    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = Variable(input).cuda()
            target = Variable(target).cuda()

            # if model.__class__.__name__ == "NetworkCIFAR":
            #    logits, _ = model(input)
            # else:
            #     logits, _ = model(input, discrete=model.architect_type in ["snas", "fixed"])
            #     TODO: change to also eval best discrete architecture
            logits, _ = model(input, **{"discrete": discrete})
            loss = criterion(logits, target)

            prec1, prec5 = accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % report_freq == 0:
                logging.info("valid %03d %e %f %f", step, objs.avg, top1.avg, top5.avg)
    # if arch is not None:
    #    model.alphas_normal.data.copy_(normal_orig)
    #    model.alphas_reduce.data.copy_(reduce_orig)

    return top1.avg, objs.avg


def copy_code_to_experiment_dir(code, experiment_dir):
    scripts_dir = os.path.join(experiment_dir, "scripts")

    if not os.path.exists(scripts_dir):
        os.makedirs(scripts_dir)
    copy_tree(code, scripts_dir)


def create_exp_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    print("Experiment dir : {}".format(path))
