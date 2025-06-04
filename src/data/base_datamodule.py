import torch
from torch.utils.data import random_split
from jax.tree_util import tree_map
import numpy as np
from ..utils.utils import logger


def numpy_collate(batch):
    return tree_map(np.asarray, torch.utils.data.default_collate(batch))


class NumpyLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        generator=None,
    ):
        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            generator=generator,
        )


class BaseDataModule:
    def __init__(self, config, generator, dataset_cls, train_transform, test_transform):
        self.root = config["dataset"]["root"]
        self.batch_size = config["dataset"]["batch_size"]
        self.num_workers = config["dataset"]["num_workers"]
        self.val_split = config["dataset"]["val_split"]
        self.forget_split = config["dataset"]["forget_split"]
        self.generator = generator
        self.split_generator = torch.Generator().manual_seed(config["dataset"]["seed"])
        self.dataset_cls = dataset_cls
        self.train_transform = train_transform
        self.test_transform = test_transform

    def setup(self):
        full_train_dataset = self.dataset_cls(
            self.root, train=True, download=True, transform=self.train_transform
        )

        val_size = int(len(full_train_dataset) * self.val_split)
        train_size = len(full_train_dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(
            full_train_dataset,
            [train_size, val_size],
            generator=self.split_generator,
        )

        forget_size = int(train_size * self.forget_split)
        retain_size = train_size - forget_size
        self.retain_dataset, self.forget_dataset = random_split(
            self.train_dataset,
            [retain_size, forget_size],
            generator=self.split_generator,
        )

        self.test_dataset = self.dataset_cls(
            self.root, train=False, download=True, transform=self.test_transform
        )
        logger.info(f"{len(self.train_dataset)=}")
        logger.info(f"{len(self.val_dataset)=}")
        logger.info(f"{len(self.test_dataset)=}")
        logger.info(f"{len(self.forget_dataset)=}")
        logger.info(f"{len(self.retain_dataset)=}")

    def train_dataloader(self):
        return NumpyLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            generator=self.generator,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def retain_dataloader(self):
        return NumpyLoader(
            self.retain_dataset,
            batch_size=self.batch_size,
            generator=self.generator,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def forget_dataloader(self):
        return NumpyLoader(
            self.forget_dataset,
            batch_size=self.batch_size,
            generator=self.generator,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return NumpyLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return NumpyLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
