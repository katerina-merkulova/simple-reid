import argparse
import datetime
import os
import os.path as osp
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

from configs.default import get_config
from data import build_dataloader
from losses import ArcFaceLoss, TripletLoss
from models import ResNet50, NormalizedClassifier
from tools.eval_metrics import evaluate
from tools.utils import AverageMeter, Logger, save_checkpoint, set_seed


def parse_option():
    parser = argparse.ArgumentParser(description='Train image-based re-id model')
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
    # Datasets
    parser.add_argument('--root', type=str, help="your root path to data directory")
    parser.add_argument('--dataset', type=str, help="market1501")
    # Miscs
    parser.add_argument('--output', type=str, help="your output path to save model and logs")
    parser.add_argument('--resume', type=str, metavar='PATH')
    parser.add_argument('--eval', action='store_true', help="evaluation only")
    parser.add_argument('--tag', type=str, help='tag for log file')
    # parser.add_argument('--gpu', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

    args, unparsed = parser.parse_known_args()
    config = get_config(args)

    return config


def main(config):
    # os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU

    sys.stdout = Logger(osp.join('logs/', 'log_train.txt'))

    # Set random seed
    set_seed(0)

    # Build dataloader
    trainloader, queryloader, galleryloader, num_classes = build_dataloader(config)
    # Build model
    model = ResNet50()
    classifier = NormalizedClassifier(num_classes)
    # Build classification and pairwise loss
    criterion_cla = ArcFaceLoss(scale=16., margin=0.1)
    criterion_pair = TripletLoss(margin=0.3, distance='cosine')
    # Build optimizer
    parameters = list(model.parameters()) + list(classifier.parameters())
    optimizer = optim.Adam(parameters, lr=0.00035, weight_decay=5e-4)
    # Build lr_scheduler
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)

    start_epoch = 0

    model = nn.DataParallel(model)
    classifier = nn.DataParallel(classifier)

    start_time = time.time()
    train_time = 0
    best_rank1 = -np.inf
    best_epoch = 0
    print("==> Start training")
    for epoch in range(start_epoch, 10):
        start_train_time = time.time()
        train(epoch, model, classifier, criterion_cla, criterion_pair, optimizer, trainloader)
        train_time += round(time.time() - start_train_time)        
        
        if (epoch+1) > 0 and 5 > 0 and \
            (epoch+1) % 5 == 0 or (epoch+1) == 10:
            print("==> Test")
            rank1 = test(model, queryloader, galleryloader)
            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch + 1

            state_dict = model.module.state_dict()
            save_checkpoint({
                'state_dict': state_dict,
                'rank1': rank1,
                'epoch': epoch,
            }, is_best, osp.join('logs/', 'checkpoint_ep' + str(epoch+1) + '.pth.tar'))
        scheduler.step()

    print(f"==> Best Rank-1 {best_rank1:.1}, achieved at epoch {best_epoch}")

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))


def train(epoch, model, classifier, criterion_cla, criterion_pair, optimizer, trainloader):
    batch_cla_loss = AverageMeter()
    batch_pair_loss = AverageMeter()
    corrects = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    classifier.train()

    end = time.time()
    for batch_idx, (imgs, pids, _) in enumerate(trainloader):
        # Measure data loading time
        data_time.update(time.time() - end)
        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward
        features = model(imgs)
        outputs = classifier(features)
        _, preds = torch.max(outputs.data, 1)
        # Compute loss
        cla_loss = criterion_cla(outputs, pids)
        pair_loss = criterion_pair(features, pids)
        loss = cla_loss + pair_loss     
        # Backward + Optimize
        loss.backward()
        optimizer.step()
        # statistics
        corrects.update(torch.sum(preds == pids.data).float()/pids.size(0), pids.size(0))
        batch_cla_loss.update(cla_loss.item(), pids.size(0))
        batch_pair_loss.update(pair_loss.item(), pids.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print(f'Epoch{epoch+1} '
          f'Time:{batch_time.sum:.1}s '
          f'Data:{data_time.sum:.1}s '
          f'ClaLoss:{batch_cla_loss.avg:.4%} '
          f'PairLoss:{batch_pair_loss.avg:.4%} '
          f'Acc:{corrects.avg:.2%} ')


def fliplr(img):
    ''' flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)

    return img_flip


@torch.no_grad()
def extract_feature(model, dataloader):
    features, pids, camids = [], [], []
    for batch_idx, (imgs, batch_pids, batch_camids) in enumerate(dataloader):
        flip_imgs = fliplr(imgs)
        batch_features = model(imgs).data
        batch_features_flip = model(flip_imgs).data
        batch_features += batch_features_flip

        features.append(batch_features)
        pids.append(batch_pids)
        camids.append(batch_camids)
    features = torch.cat(features, 0)
    pids = torch.cat(pids, 0).numpy()
    camids = torch.cat(camids, 0).numpy()

    return features, pids, camids


def test(model, queryloader, galleryloader):
    since = time.time()
    model.eval()
    # Extract features for query set
    qf, q_pids, q_camids = extract_feature(model, queryloader)
    print(f"Extracted features for query set, obtained {qf.shape} matrix")
    # Extract features for gallery set
    gf, g_pids, g_camids = extract_feature(model, galleryloader)
    print(f"Extracted features for gallery set, obtained {gf.shape} matrix")
    time_elapsed = time.time() - since
    print('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # Compute distance matrix between query and gallery
    m, n = qf.size(0), gf.size(0)
    distmat = torch.zeros((m,n))
    # Cosine similarity
    qf = F.normalize(qf, p=2, dim=1)
    gf = F.normalize(gf, p=2, dim=1)
    for i in range(m):
        distmat[i] = - torch.mm(qf[i:i+1], gf.t())
    distmat = distmat.numpy()

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    print("Results ----------------------------------------")
    print(f'top1:{cmc[0]:.1%} top5:{cmc[4]:.1%} top10:{cmc[9]:.1%} mAP:{mAP:.1%}')
    print("------------------------------------------------")

    return cmc[0]


if __name__ == '__main__':
    config = parse_option()
    main(config)