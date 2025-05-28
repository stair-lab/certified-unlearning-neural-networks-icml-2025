import flax.linen as nn


class TwoLayerNN(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(self, x, train: bool = True, mutable=None):
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=5)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.num_classes)(x)
        return x


class ThreeLayerNN(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(self, x, train: bool = True, mutable=None):
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=32)(x)
        x = nn.relu(x)
        x = nn.Dense(features=32)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.num_classes)(x)
        return x
