import flax.linen as nn


class TinyNet(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(self, x, train: bool = True, mutable=None):
        x = x.reshape(x.shape[0], -1)
        x = nn.Dense(features=5)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.num_classes)(x)
        return x

class TinierNetCIFAR10(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(self, x, train: bool = True, mutable=None):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=10)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.num_classes)(x)
        return x


class CIFAR10TinykNet(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(self, x, train: bool = True):
        he_init = nn.initializers.he_normal()
        x = nn.Conv(
            features=32, kernel_size=(3, 3), padding="same", kernel_init=he_init
        )(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(
            features=64, kernel_size=(3, 3), padding="same", kernel_init=he_init
        )(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.mean(axis=(1, 2))  # Reduces spatial dimensions to 1x1
        x = nn.Dense(self.num_classes, kernel_init=he_init)(x)
        return x


class TinierNet(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(self, x, train: bool = True, mutable=None):
        # x = nn.avg_pool(x, window_shape=(3, 3), strides=(3, 3))
        x = x.reshape(x.shape[0], -1)
        x = nn.Dense(features=20)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.num_classes)(x)
        return x
