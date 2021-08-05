# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Market shard descriptor."""

from pathlib import Path
import re

import numpy as np
from PIL import Image

from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor


class MarketShardDescriptor(ShardDescriptor):
    """Shard descriptor class."""

    def __init__(self, data_folder: str = 'Market',
                 rank_worldsize: str = '1,1') -> None:
        """Initialize MarketShardDescriptor."""
        super().__init__()

         # Settings for sharding the dataset
        self.rank_worldsize = tuple(int(num) for num in rank_worldsize.split(','))

        self.pattern = re.compile(r'([-\d]+)_c(\d)')
        self.dir_path = list(Path.cwd().parent.rglob('**/Market'))[0]
        train_path = self.dir_path / 'bounding_box_train'
        self.imgs_path = list(train_path.glob('*.jpg'))[self.rank_worldsize[0] - 1::self.rank_worldsize[1]]

    def __len__(self):        
        return len(self.imgs_path)

    def __getitem__(self, index: int):
        """Return a item by the index."""
        img_path = self.imgs_path[index]
        pid, _ = map(int, self.pattern.search(img_path.name).groups())
        
        img = Image.open(img_path)
        img = np.asarray(img)
        return img, pid

    @property
    def sample_shape(self):
        """Return the sample shape info."""
        return ['64', '128', '3']

    @property
    def target_shape(self):
        """Return the target shape info."""
        return ['1501']

    @property
    def dataset_description(self) -> str:
        """Return the dataset description."""
        return f'Market dataset, shard number {self.rank_worldsize[0]}' \
               f' out of {self.rank_worldsize[1]}'
