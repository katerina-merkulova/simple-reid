# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import tqdm
from torch.nn import init, Parameter

from openfl.federated import PyTorchTaskRunner
from openfl.utilities import TensorKey, Metric, split_tensor_dict_for_holdouts

from .losses import ArcFaceLoss, TripletLoss
from .tools import AverageMeter, evaluate, fliplr


class ResNet50(PyTorchTaskRunner):
    """ResNet50 with arcface loss for Re-Id"""

    def __init__(self, device='cpu', **kwargs):
        """Initialize.

        Args:
            data: The data loader class
            device: The hardware device to use for training (Default = 'cpu')
            **kwargs: Additional arguments to pass to the function

        """
        super().__init__(**kwargs)

#         self.num_classes = self.data_loader.dataset.num_train_pids + self.data_loader.dataset.num_query_pids
        self.num_classes = 1501
        self.device = torch.device(f'cuda:{self.data_loader.dataset.device}')
        self.init_network(device=self.device, **kwargs)
        self.classifier = NormalizedClassifier(self.num_classes, self.device)

        self.criterion_cla = ArcFaceLoss(scale=16., margin=0.1)    # self.loss_fn
        self.criterion_pair = TripletLoss(margin=0.3, distance='cosine')    # self.loss_fn
        self.param = list(self.parameters()) + list(self.classifier.parameters())
        self.optimizer = optim.Adam(self.param, lr=0.00035, weight_decay=5e-4)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[20, 40], gamma=0.1)

        self.initialize_tensorkeys_for_functions()

    @torch.no_grad()
    def extract_feature(self, dataloader):
        features, pids, camids = [], [], []
        for batch_idx, (imgs, batch_pids, batch_camids) in enumerate(dataloader):
            flip_imgs = fliplr(imgs)
            imgs, flip_imgs = imgs.to(self.device), flip_imgs.to(self.device)
            batch_features = self(imgs).data
            batch_features_flip = self(flip_imgs).data
            batch_features += batch_features_flip

            features.append(batch_features)
            pids.append(batch_pids)
            camids.append(batch_camids)
        features = torch.cat(features, 0)
        pids = torch.cat(pids, 0).numpy()
        camids = torch.cat(camids, 0).numpy()

        return features, pids, camids

    def forward(self, x):
        """Forward pass of the model.

        Args:
            x: Data input to the model for the forward pass
        """
        torch.cuda.empty_cache()
        
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        f = self.bn(x)

        return f

    def init_network(self,
                     device,
                     print_model=False,
                     **kwargs):
        """Create the network (model).

        Args:
            device: The hardware device to use for training
            print_model (bool): Print the model topology (Default=True)
            **kwargs: Additional arguments to pass to the function

        """
        resnet50 = torchvision.models.resnet50(pretrained=True)
        resnet50.layer4[0].conv2.stride = (1, 1)
        resnet50.layer4[0].downsample[0].stride = (1, 1)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.bn = nn.BatchNorm1d(2048)
        init.normal_(self.bn.weight.data, 1.0, 0.02)
        init.constant_(self.bn.bias.data, 0.0)

        self.to(device)

    def reset_opt_vars(self):
        """Reset optimizer variables.

        Resets the optimizer state variables.

        """
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)

    def train_batches(self, col_name, round_num, input_tensor_dict,
                      num_batches=None, use_tqdm=False, **kwargs):
        """Train batches.
        Train the model on the requested number of batches.
        Args:
            col_name:            Name of the collaborator
            round_num:           What round is it
            input_tensor_dict:   Required input tensors (for model)
            num_batches:         The number of batches to train on before
                                 returning
            use_tqdm (bool):     Use tqdm to print a progress bar (Default=True)
        Returns:
            global_output_dict:  Tensors to send back to the aggregator
            local_output_dict:   Tensors to maintain in the local TensorDB
        """
        self.rebuild_model(round_num, input_tensor_dict)
        # set to "training" mode
        self.train()
        self.to(self.device)
        self.classifier.train()
        self.classifier.to(self.device)

        loader = self.data_loader.get_train_loader()
        if use_tqdm:
            loader = tqdm.tqdm(loader, desc='train epoch')
        metrics = self.train_epoch(loader)
        # Output metrics tensors (scalar)
        origin = col_name
        tags = ('trained',)
        output_metric_dict = {
            TensorKey(
                metric.name, origin, round_num, True, ('metric',)
            ): metric.value
            for metric in metrics
        }

        # output model tensors (Doesn't include TensorKey)
        output_model_dict = self.get_tensor_dict(with_opt_vars=True)
        global_model_dict, local_model_dict = split_tensor_dict_for_holdouts(
            self.logger, output_model_dict,
            **self.tensor_dict_split_fn_kwargs
        )

        # Create global tensorkeys
        global_tensorkey_model_dict = {
            TensorKey(tensor_name, origin, round_num, False, tags):
                nparray for tensor_name, nparray in global_model_dict.items()
        }
        # Create tensorkeys that should stay local
        local_tensorkey_model_dict = {
            TensorKey(tensor_name, origin, round_num, False, tags):
                nparray for tensor_name, nparray in local_model_dict.items()
        }
        # The train/validate aggregated function of the next round will look
        # for the updated model parameters.
        # This ensures they will be resolved locally
        next_local_tensorkey_model_dict = {
            TensorKey(tensor_name, origin, round_num + 1, False, ('model',)): nparray
            for tensor_name, nparray in local_model_dict.items()}

        global_tensor_dict = {
            **output_metric_dict,
            **global_tensorkey_model_dict
        }
        local_tensor_dict = {
            **local_tensorkey_model_dict,
            **next_local_tensorkey_model_dict
        }

        # Update the required tensors if they need to be pulled from the
        # aggregator
        # TODO this logic can break if different collaborators have different
        # roles between rounds.
        # For example, if a collaborator only performs validation in the first
        # round but training in the second, it has no way of knowing the
        # optimizer state tensor names to request from the aggregator because
        # these are only created after training occurs. A work around could
        # involve doing a single epoch of training on random data to get the
        # optimizer names, and then throwing away the model.
        if self.opt_treatment == 'CONTINUE_GLOBAL':
            self.initialize_tensorkeys_for_functions(with_opt_vars=True)

        # This will signal that the optimizer values are now present,
        # and can be loaded when the model is rebuilt
        self.train_round_completed = True

        # Return global_tensor_dict, local_tensor_dict
        return global_tensor_dict, local_tensor_dict

    def train_epoch(self, trainloader):
        """Train single epoch.
        Override this function in order to use custom training.
        Args:
            trainloader: Train dataset batch generator. Yields (samples, targets) tuples of
            size = `self.data_loader.batch_size`.
        Returns:
            Metric: An object containing name and np.ndarray value.
        """

        batch_cla_loss = AverageMeter()
        batch_pair_loss = AverageMeter()
        corrects = AverageMeter()

        for batch_idx, (imgs, pids, _) in enumerate(trainloader):
            imgs, pids = torch.tensor(imgs).to(self.device), torch.tensor(pids).to(self.device)
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            # Forward
            features = self(imgs)
            outputs = self.classifier(features)
            _, preds = torch.max(outputs.data, 1)
            # Compute loss
            cla_loss = self.criterion_cla(outputs, pids)
            print(f'{features.shape}')
            pair_loss = self.criterion_pair(features, pids)
            loss = cla_loss + pair_loss
            # Backward + Optimize
            loss.backward()
            self.optimizer.step()
            # statistics
            corrects.update(torch.sum(preds == pids.data).float() / pids.size(0), pids.size(0))
            batch_cla_loss.update(cla_loss.item(), pids.size(0))
            batch_pair_loss.update(pair_loss.item(), pids.size(0))

        self.scheduler.step()
        
        logs = open('/home/merkulov/federated_project/market_simple_re-id/old_openfl/logs.txt', 'a')
        print(f'ClaLoss:{batch_cla_loss.avg:.2f} '
              f'PairLoss:{batch_pair_loss.avg:.2f} '
              f'Acc:{corrects.avg:.2%} ', file=logs)
        logs.close()
        
        return (
            Metric(name='Accuracy', value=np.array(corrects.avg.cpu())),
            Metric(name='ArcFaceLoss', value=np.array(batch_cla_loss.avg)),
            Metric(name='TripletLoss', value=np.array(batch_pair_loss.avg))
        )

    def validate(self, col_name, round_num, input_tensor_dict, use_tqdm=False, **kwargs):
        """Validate.

        Run validation of the model on the local data.

        Args:
            col_name:            Name of the collaborator
            round_num:           What round is it
            input_tensor_dict:   Required input tensors (for model)
            use_tqdm (bool):     Use tqdm to print a progress bar (Default=True)

        Returns:
            global_output_dict:  Tensors to send back to the aggregator
            local_output_dict:   Tensors to maintain in the local TensorDB

        """
        self.rebuild_model(round_num, input_tensor_dict, validation=True)
        self.eval()

        # Extract features for query set
        query_loader = self.data_loader.get_query_loader()
        if use_tqdm:
            query_loader = tqdm.tqdm(query_loader, desc='query')
        qf, q_pids, q_camids = self.extract_feature(query_loader)

        # Extract features for gallery set
        gallery_loader = self.data_loader.get_gallery_loader()
        if use_tqdm:
            gallery_loader = tqdm.tqdm(gallery_loader, desc='gallery')
        gf, g_pids, g_camids = self.extract_feature(gallery_loader)

        # Compute distance matrix between query and gallery
        m, n = qf.size(0), gf.size(0)
        distmat = torch.zeros((m, n))
        # Cosine similarity
        qf = F.normalize(qf, p=2, dim=1)
        gf = F.normalize(gf, p=2, dim=1)
        for i in range(m):
            distmat[i] = - torch.mm(qf[i:i + 1], gf.t())
        distmat = distmat.numpy()

        cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

        origin = col_name
        suffix = 'validate'
        if kwargs['apply'] == 'local':
            suffix += '_local'
        else:
            suffix += '_agg'
        tags = ('metric', suffix)
        # TODO figure out a better way to pass
        #  in metric for this pytorch validate function
        output_tensor_dict = {
            TensorKey('top1', origin, round_num, True, tags):
               cmc[0] * 100,
            TensorKey('top5', origin, round_num, True, tags):
               cmc[4] * 100,
            TensorKey('top10', origin, round_num, True, tags):
               cmc[9] * 100,
            TensorKey('mAP', origin, round_num, True, tags):
                mAP * 100
        }

        logs = open('/home/merkulov/federated_project/market_simple_re-id/old_openfl/logs.txt', 'a')
        print(f'Results for Epoch {round_num + 1} Collaborator {col_name} {suffix}', file=logs)
        print(f'top1:{cmc[0]:.1%} top5:{cmc[4]:.1%} top10:{cmc[9]:.1%} mAP:{mAP:.1%}', file=logs)
        print('------------------------------------------------', file=logs)
        logs.close()
        
        # Empty list represents metrics that should only be stored locally
        return output_tensor_dict, {}


class NormalizedClassifier(nn.Module):
    """Simple CNN for classification."""

    def __init__(self, num_classes, device):
        """Initialize.

        Args:
            data: The data loader class
            device: The hardware device to use for training (Default = "cpu")
            **kwargs: Additional arguments to pass to the function
        """
        super().__init__()

        self.weight = Parameter(torch.Tensor(num_classes, 2048))
        self.weight.data.uniform_(-1, 1).renorm_(2,0,1e-5).mul_(1e5)

    def forward(self, x):
        """Forward pass of the model.

        Args:
            x: Data input to the model for the forward pass
        """
        w = self.weight

        x = F.normalize(x, p=2, dim=1)
        w = F.normalize(w, p=2, dim=1)

        return F.linear(x, w)
