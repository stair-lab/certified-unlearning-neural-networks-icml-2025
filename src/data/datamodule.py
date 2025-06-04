from src.data.cifar_dataset import CIFAR10DataModule
from src.data.mnist_dataset import MNISTDataModule

def get_datamodule(config, generator):
    if config["dataset"]["name"] == "cifar10":
        return CIFAR10DataModule(config, generator)
    elif config["dataset"]["name"] == "mnist":
        return MNISTDataModule(config, generator)
    elif config["dataset"]["name"] == "cifar10_feature":
        from src.data.cifar_dataset import CIFAR10FeatureDataModule
        return CIFAR10FeatureDataModule(config, generator)
    elif config["dataset"]["name"] == "cifar100_feature":
        from src.data.cifar_dataset import CIFAR100FeatureDataModule
        return CIFAR100FeatureDataModule(config, generator)
    else:
        raise ValueError("Unknown dataset")
