# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""
from torch.utils.data import DataLoader

import code.transforms as T
from .tools import ImageDataset, RandomIdentitySampler
from .datasets import Market1501

from openfl.federated import PyTorchDataLoader


class PyTorchMarket(PyTorchDataLoader):
    """PyTorch data loader for MNIST dataset."""

    def __init__(self, data_path, batch_size, **kwargs):

        """
        Instantiate the data object.
        Args:
            batch_size: Size of batches used for all data loaders
            kwargs: consumes all un-used kwargs
        Returns:
            None
        """
        super().__init__(batch_size, **kwargs)

        self.dataset = Market1501(root=data_path, split_id=0,
                                  cuhk03_labeled=False,
                                  cuhk03_classic_split=False)

        self.transform_train = T.Compose([
            T.RandomCroping(256, 128, p=0.5),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.RandomErasing(probability=0.5)
        ])
        self.transform_test = T.Compose([
            T.Resize((265, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.batch_size = batch_size
        self.X_train = None
        self.y_train = None
        self.X_valid = None
        self.y_valid = None

    def get_feature_shape(self):
        """Get the shape of an example feature array.
        Returns:
            tuple: shape of an example feature array
        """
        return self.dataset.num_train_pids

    def get_train_loader(self):
        """
        Get training data loader.
        Returns
        -------
        loader object
        """
        return DataLoader(ImageDataset(self.dataset.train, transform=self.transform_train),
                          sampler=RandomIdentitySampler(self.dataset.train, num_instances=4),
                          batch_size=64, num_workers=4,
                          pin_memory=True, drop_last=True)

    def get_query_loader(self):
        """
        Get query data loader.
        Returns:
            loader object
        """
        return DataLoader(ImageDataset(self.dataset.train, transform=self.transform_test),
                          batch_size=512, num_workers=4,
                          pin_memory=True, drop_last=False, shuffle=False)

    def get_gallery_loader(self):
        """
        Get gallery data loader.
        Returns:
            loader object
        """
        return DataLoader(ImageDataset(self.dataset.gallery, transform=self.transform_test),
                          batch_size=512, num_workers=4,
                          pin_memory=True, drop_last=False, shuffle=False)

    def get_train_data_size(self):
        """
        Get total number of training samples.
        Returns:
            int: number of training samples
        """
        raise NotImplementedError

    def get_valid_data_size(self):
        """
        Get total number of validation samples.
        Returns:
            int: number of validation samples
        """
        raise NotImplementedError
