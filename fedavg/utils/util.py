from typing import Tuple, List
import torch
import cloudpickle
import jax
import os
import errno

import numpy as np
import jax.numpy as jnp
from jaxlib.xla_client import Shape
from torch.utils.data.dataloader import DataLoader

from flax import serialization


@jax.jit
def ball_get_random(rng, inputs, r):
    """ Get random point with shape in a ball of radius r. """
    d = jax.random.normal(rng, inputs.shape)
    norms = jnp.linalg.norm(d.reshape(inputs.shape[0], -1), axis=1)
    norms = norms.reshape([norms.shape[0]] + [1] * (len(inputs.shape)-1))
    rng, uni_rng = jax.random.split(rng)
    scale = r / norms * jax.random.uniform(uni_rng, norms.shape)
    # return d * r / norms
    return d * scale


def one_hot(x, k, dtype=jnp.float32):
    """Create a one-hot encoding of x of size k."""
    return jnp.array(x[:, None] == jnp.arange(k), dtype)


# TODO Allow a range/batch of images
def get_image_from_loader(loader: DataLoader, idx: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """ Gets a sub-batch of images from the loader. NOTE: This does not any loader specific transformations

    Args:
        loader (any): The corresponding dataloader
        idx (List[int]): List of indices that we want to take a look at

    Returns:
        Tuple[np.ndarray, np.ndarray]: The corresponding input and target tuple
    """
    dataset = loader.dataset
    return dataset[idx]


def generate_init_img(rng, shape, att_init):
    if att_init == 'normal':
        at_img = jax.random.normal(rng, shape)
    if att_init == 'uniform':
        at_img = jax.random.uniform(rng, shape, minval=-1.0, maxval=1.0)
    elif att_init == 'zero':
        at_img = jnp.zeros(shape)
    
    return at_img


def load_model_flax(path, net, dummy_input, net_state, defense_state=None, init_on_failure=True, args=None):
    with open(path, "rb") as file:
        states_dict = cloudpickle.load(file)
        net_state = serialization.from_state_dict(net_state, states_dict['net'])
        defense_state = serialization.from_state_dict( defense_state, states_dict['defense'] ) if defense_state is None else defense_state
        print("Succesfully loaded model state")
        return net_state, defense_state


def store_model_flax(path, net_state, defense_state=None, generate_path=False):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    with open(path, "wb") as out_file:
        net_dict_out = serialization.to_state_dict(net_state)
        defense_dict_out = serialization.to_state_dict(defense_state)
        cloudpickle.dump({'net': net_dict_out, 'defense': defense_dict_out}, out_file)
        print("Succesfully stored model parameters at: ", path)


def load_model_flax_fed(path, net, dummy_input, net_state, defense_state=None, client_id=None, args=None):
    with open(path, "rb") as file:
        states_dict = cloudpickle.load(file)
        net_state = serialization.from_state_dict(net_state, states_dict['net'])
        if defense_state is not None:
            defense_state = serialization.from_state_dict(defense_state, states_dict['defense'][client_id])
        print("Succesfully loaded model state")
        return net_state, defense_state


def store_model_flax_fed(path, net_state, defense_states=None, generate_path=False):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    with open(path, "wb") as out_file:
        net_dict_out = serialization.to_state_dict(net_state)
        defense_dict_out = [serialization.to_state_dict(defense_state) for defense_state in defense_states]
        cloudpickle.dump({'net': net_dict_out, 'defense': defense_dict_out}, out_file)
        print("Succesfully stored model parameters at: ", path)
