import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
import copy

from models import MixNet
from models import mixnet_builder


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DATA_PATH', default='./data/',
                    help='path to imagenet data (default: ./data/)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='mixnet_s',
                    help='model architecture (default: resnet18)')
parser.add_argument('--dtype', type=str, default="cifar10")
parser.add_argument('--scheduler', type=str, default='exp', help="Learning rate scheduler type")
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--eps', type=float, default=0.001)
parser.add_argument('--wd', '--weight-decay', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-5)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--image_size', default=32, type=int,
                    help='image size')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--tmp_data_dir', type=str, default='/tmp/cache/', help='temp data dir')

def get_scheduler(optim, sche_type, step_size, t_max):
    if sche_type == "exp":
        return StepLR(optim, step_size, 0.97)
    elif sche_type == "cosine":
        return CosineAnnealingLR(optim, t_max)
    else:
        return None

def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    global args
    args = parser.parse_args()
    args.save = 'eval-{}-{}-{}'.format(args.save, args.dtype,time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    logging.info(("=> load data '{}'".format(args.dtype)))
    if args.dtype == 'cifar10':
        train_transform, valid_transform = utils._data_transforms_cifar10(args, cutout=True)
        train_data = dset.CIFAR10(root=args.tmp_data_dir, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR10(root=args.tmp_data_dir, train=False, download=True, transform=valid_transform)
        num_classes = 10
        update_lrs = [150, 250, 350]
    elif args.dtype == 'cifar100':
        train_transform, valid_transform = utils._data_transforms_cifar100(args, cutout=True)
        train_data = dset.CIFAR100(root=args.tmp_data_dir, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR100(root=args.tmp_data_dir, train=False, download=True, transform=valid_transform)
        num_classes = 100
        update_lrs = [40, 80, 160, 300]
    else:
        logging.info('no data type available')
        sys.exit(1)
    logging.info("update lrs: '{}'".format(update_lrs))
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)

    logging.info(("=> creating model '{}'".format(args.arch)))
    blocks_args, global_params = mixnet_builder.get_model_params(args.arch)
    model = MixNet(input_size=32, num_classes=num_classes, blocks_args=blocks_args, global_params=global_params)
    # print(model)
    # exit(0)
    logging.info("args = %s", args)
    # logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    logging.info("param size = %fMB", model._num_params / 1e6)

    # exit(0)
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            # model.make_cuda_and_parallel()
            # model.avgpool = torch.nn.DataParallel(model.avgpool)
            # model.classifier = torch.nn.DataParallel(model.classifier)
            model = torch.nn.DataParallel(model)
            model = model.cuda()
            

    criterion = nn.CrossEntropyLoss().cuda()
    
    if args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, momentum=args.momentum, eps=args.eps, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    cudnn.benchmark = True    
    # scaled_lr = args.lr * args.batch_size / 256
    # optim = {
    #     "adam" : lambda : torch.optim.Adam(model.parameters()),
    #     "rmsprop" : lambda : torch.optim.RMSprop(model.parameters(), lr=scaled_lr, momentum=args.momentum, eps=args.eps, weight_decay=args.weight_decay)
    # }[args.optim]()

    # scheduler = get_scheduler(optim, args.scheduler, int(2.4*len(train_queue)), args.epochs * len(train_queue))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    best_acc = 0.0
    cur_lr = args.lr
    for epoch in range(args.epochs):
        scheduler.step()
        logging.info('Epoch: %d lr %e', epoch, scheduler.get_lr()[0])
        # cur_lr = adjust_learning_rate(optimizer, epoch, cur_lr, update_lrs)
        # logging.info('Epoch: %d lr %e', epoch, cur_lr)
        start_time = time.time()
        train_acc, train_obj = train(train_queue, model, criterion, optimizer)
        logging.info('Train_acc: %f', train_acc)
        valid_acc, valid_obj = test(valid_queue, model, criterion)
        if valid_acc > best_acc:
            best_acc = valid_acc
        logging.info('Valid_acc: %f', valid_acc)
        end_time = time.time()
        duration = end_time - start_time
        print('Epoch time: %ds.' % duration )
        utils.save(model, os.path.join(args.save, 'weights.pt'))

def adjust_learning_rate(optimizer, epoch, cur_lr, update_lrs):
    if epoch in update_lrs:
        cur_lr = cur_lr * 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr
    
    return cur_lr

def train(train_queue, model, criterion, optimizer):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    model.train()
    
    for step, (input, target) in enumerate(train_queue):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        prec1, _ = utils.accuracy(logits, target, topk=(1,5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('Train Step: %03d Objs: %e Acc: %f', step, objs.avg, top1.avg)
    
    return top1.avg, objs.avg

def test(valid_queue, model, criterion):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            logits = model(input)
            loss = criterion(logits, target)
        
        prec1, _ = utils.accuracy(logits, target, topk=(1,5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('Valid Step: %03d Objs: %e Acc: %f', step, objs.avg, top1.avg)

    return top1.avg, objs.avg

if __name__== '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    duration = end_time - start_time
    logging.info('Eval time: %ds.', duration)