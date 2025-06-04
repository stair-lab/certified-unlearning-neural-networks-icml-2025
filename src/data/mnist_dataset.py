from torchvision import transforms
from torchvision.datasets import MNIST

from src.data.base_datamodule import BaseDataModule


class MNISTDataModule(BaseDataModule):
    def __init__(self, config, generator):
        augmentations = [
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
        ]
        normalize = [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.permute(1, 2, 0)), # Pytorch uses channels first, but we want channels last
        ]
        train_transform = transforms.Compose(augmentations + normalize)
        test_transform = transforms.Compose(normalize)
        super().__init__(config, generator, MNIST, train_transform, test_transform)
