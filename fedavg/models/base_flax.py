from flax import linen as nn
from typing import Sequence


class MLP_Flax(nn.Module):
    n_targets: int
    sizes: Sequence[int]

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        for i, sz in enumerate(self.sizes):
            x = nn.Dense(features=sz)(x)
            x = nn.relu(x)
        self.sow( 'label_pred', 'last_relu', x )
        x = nn.Dense(features=self.n_targets, name='last_layer')(x)
        x = nn.log_softmax(x)
        return x


class CNN(nn.Module):
    """A simple CNN model."""
    n_targets: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=100)(x)
        x = nn.relu(x)
        self.sow( 'label_pred', 'last_relu', x )
        x = nn.Dense(features=self.n_targets, name='last_layer')(x)
        x = nn.log_softmax(x)
        return x

class CNN_3(nn.Module):
    """A simple CNN model."""
    n_targets: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=128, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=200)(x)
        x = nn.relu(x)
        self.sow( 'label_pred', 'last_relu', x )
        x = nn.Dense(features=self.n_targets, name='last_layer')(x)
        x = nn.log_softmax(x)
        return x




class CNN_2(nn.Module):
    """A simple CNN model."""
    n_targets: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=128, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        self.sow( 'label_pred', 'last_relu', x )
        x = nn.Dense(features=self.n_targets, name='last_layer')(x)
        x = nn.log_softmax(x)
        return x


class ConvBig(nn.Module):
    """
    def ConvBig(c, **kargs):
         return n.LeNet([ (32,3,3,1), (32,4,4,2) , (64,3,3,1), (64,4,4,2)], [512, 512,c], padding = 1, last_lin = True, last_zono = True, **kargs)
    """
    n_targets: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(1, 1))(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1))(x)
        x = nn.relu(x)
        # x = nn.Conv(features=64//2, kernel_size=(4, 4), strides=(2, 2))(x)
        # x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        # x = nn.Dense(features=512)(x)
        # x = nn.relu(x)
        x = nn.Dense(features=512)(x)
        x = nn.relu(x)
        self.sow( 'label_pred', 'last_relu', x )
        x = nn.Dense(features=self.n_targets, name='last_layer')(x)
        x = nn.log_softmax(x)
        return x


def get_flax_network(name, n_targets):
    split = name.split("_")
    net_name = split[0]

    if net_name == 'cnn':
        return CNN(n_targets=n_targets)
    elif net_name == 'cnn2':
        return CNN_2(n_targets=n_targets)
    elif net_name == 'cnn3':
        return CNN_3(n_targets=n_targets)
    elif net_name == 'convbig':
        return ConvBig(n_targets=n_targets)
    elif net_name == 'mlp':
        layer_list = split[1:]
        layer_list = list(map(lambda x: int(x), layer_list))
        return MLP_Flax(n_targets=n_targets, sizes=layer_list)
    else:
        assert False
