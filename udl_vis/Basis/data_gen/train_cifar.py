# GPLv3 License
# Copyright (C) 2021 , UESTC
# All Rights Reserved
#
# @Author  : Xiao Wu
# @reference:
import torch
from torch import nn
from torch import optim
import argparse
import time
from UDL.Basis.auxiliary import AverageMeter, accuracy, show_memory_info
from UDL.Basis.data_gen.DatasetFromH5 import DatasetH5

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR',default='../../path_to_imagenet',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',default=False,
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,  #os.environ["CUDA_VISIBLE_DEVICES"]的映射约束下的顺序
                    help='GPU id to use.')

args = parser.parse_args()

def train():


    train_set = DatasetH5("../00-data/cifar10_train_chunks.h5")
    # val_set = Dataset_H5("cifar_val_chunks.h5")

    train_loader = DataLoader(train_set, num_workers=8, shuffle=True, batch_size=64)
    # train_loader = MultiEpochsDataLoader(train_set, batch_size=args.batch_size,
    #                                shuffle=True, num_workers=args.workers,
    #                                pin_memory=False)
    # if torch.cuda.is_available():
    #     train_loader = CudaDataLoader(train_loader, 'cuda', queue_size=4)

    net = MobileNetV2().cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)



    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    epoch_time = time.time()

    for epoch in range(args.epochs):
        net.train()
        # show_memory_info(0)
        for batch_idx, batch in enumerate(train_loader):
            # show_memory_info(0)
            data_time.update(time.time() - end)
            img, targets = batch
            img = img.cuda()
            targets = targets.cuda()
            outputs = net(img)
            loss = criterion(outputs, targets)
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), img.size(0))
            top1.update(acc1[0], img.size(0))
            top5.update(acc5[0], img.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.print_freq == 0:
                # print('[one_epoch]: [{0}/{1}]\t '
                #            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                #            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                #            .format(batch_idx, len(train_loader), batch_time=batch_time, data_time=data_time))
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      '[DALI] Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      epoch, batch_idx, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1, top5=top5))
        show_memory_info(epoch)

        print(time.time() - epoch_time)
        epoch_time = time.time()

if __name__ == "__main__":

    train()