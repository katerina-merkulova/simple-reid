import glob
import os.path as osp
from pathlib import Path
import re


DATAPATH = list(Path.cwd().parents[2].rglob('**/Market'))[0]    # parent directory of project


class Market1501(object):
    """
    Market1501

    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    URL: http://www.liangzheng.org/Project/project_reid.html
    
    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """

    def __init__(self, root='data', **kwargs):
        self.dataset_dir = Path(DATAPATH)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()

        train, num_train_pids, num_train_imgs = self._process_dir(self.train_dir, relabel=True)
        query, num_query_pids, num_query_imgs = self._process_dir(self.query_dir, relabel=False)
        gallery, num_gallery_pids, num_gallery_imgs = self._process_dir(self.gallery_dir, relabel=False)
        num_total_pids = num_train_pids + num_query_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        print(
            '=> Market1501 loaded\n'
            'Dataset statistics:\n'
            '  ------------------------------\n'
            '  subset   | # ids | # images\n'
            '  ------------------------------\n'
            f'  train    | {num_train_pids} | {num_train_imgs}\n'
            f'  query    | {num_query_pids} | {num_query_imgs}\n'
            f'  gallery  | {num_gallery_pids} | {num_gallery_imgs}\n'
            '------------------------------\n'
            f'total    | {num_total_pids} | {num_total_imgs}\n'
            '  ------------------------------'
        )

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError(f'{self.dataset_dir} is not available')
        if not osp.exists(self.train_dir):
            raise RuntimeError(f'{self.train_dir} is not available')
        if not osp.exists(self.query_dir):
            raise RuntimeError(f'{self.query_dir} is not available')
        if not osp.exists(self.gallery_dir):
            raise RuntimeError(f'{self.gallery_dir} is not available')

    def _process_dir(self, dir_path, relabel=False, label_start=0):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            if label_start == 0:
                assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1 # index starts from 0
            if relabel: pid = pid2label[pid] + label_start
            dataset.append((img_path, pid, camid))

        num_pids = len(pid_container)
        num_imgs = len(dataset)
        return dataset, num_pids, num_imgs
