import datetime
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

from data import build_dataloader
from losses import ArcFaceLoss, TripletLoss
from models import ResNet50, NormalizedClassifier
from tools.eval_metrics import evaluate
from tools.utils import AverageMeter

np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    # Build dataloader
    trainloader, queryloader, galleryloader, num_classes = build_dataloader()
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

    start_time = datetime.datetime.now().time()
    logs = open('logs.txt', 'a')
    print(f'==> Start training {start_time}', file=logs)
    logs.close()

    for epoch in range(3):
        train(epoch, model, classifier, criterion_cla, criterion_pair, optimizer, trainloader)
        scheduler.step()

    test(model, queryloader, galleryloader)

    end_time = datetime.datetime.now().time()
    logs = open('logs.txt', 'a')
    print(f'==> End training {end_time}', file=logs)
    print(f'Training time: {end_time - start_time}')
    logs.close()


def train(epoch, model, classifier, criterion_cla, criterion_pair, optimizer, trainloader):
    batch_cla_loss = AverageMeter()
    batch_pair_loss = AverageMeter()
    corrects = AverageMeter()

    model.to('cuda')
    model.train()
    classifier.to('cuda')
    classifier.train()

    for batch_idx, (imgs, pids, _) in enumerate(trainloader):
        imgs, pids = imgs.cuda(), pids.cuda()
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
        corrects.update(torch.sum(preds == pids.data).float() / pids.size(0), pids.size(0))
        batch_cla_loss.update(cla_loss.item(), pids.size(0))
        batch_pair_loss.update(pair_loss.item(), pids.size(0))

    logs = open('logs.txt', 'a')
    print(f'Epoch {epoch + 1} '
          f'ClaLoss:{batch_cla_loss.avg:.2f} '
          f'PairLoss:{batch_pair_loss.avg:.2f} '
          f'Acc:{corrects.avg:.2%} ', file=logs)
    logs.close()


def fliplr(img):
    ''' flip horizontal'''
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)

    return img_flip


@torch.no_grad()
def extract_feature(model, dataloader):
    features, pids, camids = [], [], []
    for batch_idx, (imgs, batch_pids, batch_camids) in enumerate(dataloader):
        flip_imgs = fliplr(imgs)
        imgs, flip_imgs = imgs.cuda(), flip_imgs.cuda()
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
    model.eval()
    # Extract features for query set
    qf, q_pids, q_camids = extract_feature(model, queryloader)
    print(f'Extracted features for query set, obtained {qf.shape} matrix')
    # Extract features for gallery set
    gf, g_pids, g_camids = extract_feature(model, galleryloader)
    print(f'Extracted features for gallery set, obtained {gf.shape} matrix')
    # Compute distance matrix between query and gallery
    m, n = qf.size(0), gf.size(0)
    distmat = torch.zeros((m, n))
    # Cosine similarity
    qf = F.normalize(qf, p=2, dim=1)
    gf = F.normalize(gf, p=2, dim=1)
    for i in range(m):
        distmat[i] = - torch.mm(qf[i:i + 1], gf.t())
    distmat = distmat.numpy()

    print('Computing CMC and mAP')
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    print('Results ----------------------------------------')
    print(f'top1:{cmc[0]:.1%} top5:{cmc[4]:.1%} top10:{cmc[9]:.1%} mAP:{mAP:.1%}')
    print('------------------------------------------------')

    return cmc[0]


if __name__ == '__main__':
    main()
