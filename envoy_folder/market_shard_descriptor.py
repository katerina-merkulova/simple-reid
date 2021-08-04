# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Market shard descriptor."""

from pathlib import Path
import re

from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor


class MarketShardDescriptor(ShardDescriptor):
    """Shard descriptor class."""

    def __init__(self, data_folder: str = 'Market',
                 rank_worldsize: str = '1,1') -> None:
        """Initialize MarketShardDescriptor."""
        super().__init__()

         # Settings for sharding the dataset
        self.rank_worldsize = tuple(int(num) for num in rank_worldsize.split(','))

    def __getitem__(self, index: int):
        """Return a item by the index."""
        pattern = re.compile(r'([-\d]+)_c(\d)')
        dir_path = Path('Market') / 'bounding_box_train'
        img_path = list(dir_path.glob('*.jpg'))[index]

        pid, _ = map(int, pattern.search(img_path.name).groups())
        return img_path, pid

    @property
    def sample_shape(self):
        """Return the sample shape info."""
        raise NotImplementedError

    @property
    def target_shape(self):
        """Return the target shape info."""
        raise NotImplementedError

    @property
    def dataset_description(self) -> str:
        """Return the dataset description."""
        return f'Market dataset, shard number {self.rank_worldsize[0]}' \
               f' out of {self.rank_worldsize[1]}'
