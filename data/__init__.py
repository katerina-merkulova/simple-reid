from torch.utils.data import DataLoader

import data.transforms as T
from data.dataset_loader import ImageDataset
from data.datasets import Market1501
from data.samplers import RandomIdentitySampler


def build_dataloader():
    dataset = Market1501(root='data', split_id=0,
                         cuhk03_labeled=False,
                         cuhk03_classic_split=False)

    transform_train = T.Compose([
        T.RandomCroping(256, 128, p=0.5),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(probability=0.5)
    ])
    transform_test = T.Compose([
        T.Resize((265, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    trainloader = DataLoader(ImageDataset(dataset.train, transform=transform_train),
                             sampler=RandomIdentitySampler(dataset.train, num_instances=4),
                             batch_size=64, num_workers=4,
                             pin_memory=True, drop_last=True)
    queryloader = DataLoader(ImageDataset(dataset.query, transform=transform_test),
                             batch_size=512, num_workers=4,
                             pin_memory=True, drop_last=False, shuffle=False)

    galleryloader = DataLoader(ImageDataset(dataset.gallery, transform=transform_test),
                               batch_size=512, num_workers=4,
                               pin_memory=True, drop_last=False, shuffle=False)

    return trainloader, queryloader, galleryloader, dataset.num_train_pids
