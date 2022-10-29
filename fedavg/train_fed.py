from typing import Generator, Mapping, Tuple
import os
import errno
import argparse
import csv
import cloudpickle
from datetime import datetime

import json
import jax
import jax.numpy as jnp
import jax.scipy.optimize as jso
import numpy as np
import optax
import torch
import time
import shutil

from jax.tree_util import tree_flatten, tree_unflatten

from tqdm import tqdm, trange
from datasets.common import get_dataset_by_name
from datasets.distributions import get_prior
from utils.util import load_model_flax_fed, load_model_flax, store_model_flax_fed, store_model_flax, get_image_from_loader, ball_get_random
from utils.flax_losses import flax_cross_entropy_loss, flax_compute_metrics, flax_get_train_methods, flax_get_attack_loss_and_update
from utils.plotting import visualize
from models.base import get_network
from models.base_flax import get_flax_network, MLP_Flax
from utils.measures import get_acc_metric, l2_distance, compute_noise_l2

from args_factory import get_args
from defenses.defense import get_defense
from defenses.noise_defenses import get_evaluate_nets, get_defend_grad, MIN_SIGMA, MAX_SIGMA
from attacks.grad_attack import attack_via_grad_opt_flax, compute_matching, get_att_losses
from utils.neptune_setup import get_neptune, log_neptune, log_neptune_final 

from flax import linen as nn
from flax.training import train_state

from jax.config import config as jaxconfig


def aggregate_batch_metrics(batch_metrics):
    batch_metrics_np = jax.device_get(batch_metrics)
    agg_metrics_np = {k: np.mean([metrics[k] for metrics in batch_metrics_np]) for k in batch_metrics_np[0]}
    return agg_metrics_np

def attack_flax(args, net_state=None, def_state=None, client_id=None):
    rng = jax.random.PRNGKey(0)
    rng, dataset_rng = jax.random.split(rng)

    neptune_logger = get_neptune(args)

    _, (train_loaders, test_loaders), n_targets, (dummy_input, dummy_targets) = get_dataset_by_name(
        args.dataset, args.batch_size, args.batch_size, k_batches=args.k_batches, rng=dataset_rng, n_clients=args.n_clients, maxn=args.max_batch, same_cls=args.same_cls)
    train_loader, test_loader = train_loaders[client_id], test_loaders[client_id]
    prior = get_prior(args, args.dataset, dataset_rng)
    net = get_flax_network(args.network, n_targets)
    create_train_state, _, eval_step, _  = flax_get_train_methods(net, dummy_input)
    rng, init_rng = jax.random.split(rng)
    if net_state is None:
        net_state = create_train_state(init_rng, learning_rate=args.learning_rate)

    if args.defense is not None:
        rng, defense_rng = jax.random.split(rng)
        dummy_grad = jax.grad(lambda p: flax_cross_entropy_loss(log_probs=net.apply({'params': p}, dummy_input), labels=dummy_targets))(net_state.params)
        _, def_perturb_grads, _, init_defense_params = get_defense(args.defense, defense_rng, args.batch_size, dummy_input, dummy_grad)
    else:
        def_perturb_grads, _, defense_params = None, None, None

    if args.defense is not None:
        if def_state is None:
            def_state = train_state.TrainState.create(apply_fn=net.apply, params=init_defense_params, tx=optax.adam(learning_rate=args.defense_lr))
            if args.path is not None:
                net_state, def_state = load_model_flax_fed(args.path, net, dummy_input, net_state, def_state, client_id, args=args)
                #net_state, def_state = load_model_flax(args.path, net, dummy_input, net_state, def_state, client_id, args=args)
            #net_state = load_model_flax(args.path, net, dummy_input, net_state, def_state, client_id, args=args)
            #import pdb; pdb.set_trace()
            #net_state, def_state = load_model_flax_fed(args.path, net, dummy_input, net_state, def_state, client_id, args=args)
        defense_params = def_state.params
    else:
        if args.path is not None:
            net_state, _ = load_model_flax(args.path, net, dummy_input, net_state, None, args=args)

    defend_grad, nodefend_grad = get_defend_grad(net, def_perturb_grads, args.batch_size, args.learning_rate, args.epochs, args.rand_batch)
    if args.defense is None:
        defend_grad = nodefend_grad
    batch_metrics = []
    compiled_att_funcs = get_att_losses(net, prior, args)
    tl = iter(train_loader)
    pbar = tqdm(range(args.n_attack))
    start_time = time.time()
    for i in pbar:
        if not neptune_logger is None:
            neptune_logger['logs/curr_input'].log(i)
        inputs, targets = next(tl)
        rng, att_rng = jax.random.split(rng)
        root_dir = f'{args.vis_dir}/{i}'
        args.root_dir = root_dir
        shutil.rmtree(args.root_dir, ignore_errors=True)
        os.makedirs(root_dir)
        if args.dataset == 'FEMNIST' or args.dataset == 'CelebA':
            inputs = inputs[0]
            targets = targets[0]
        if len(inputs.shape) == 4:
            inputs = np.einsum("bijk -> bjki", inputs)
        pbar.set_description(f'Batch size {inputs.shape[0]}')
        metrics = attack_via_grad_opt_flax(root_dir, compiled_att_funcs, att_rng, net, defend_grad, defense_params, net_state.params, inputs, targets, n_targets, prior, args)
        batch_metrics.append(metrics)
        metrics['batch_size'] = inputs.shape[0]
        log_neptune( args, i, neptune_logger, metrics )
        print( metrics )

    end_time = time.time()
    res_metrics_np = {
        'runtime': end_time - start_time,
    }
    log_neptune_final( neptune_logger, res_metrics_np )
    print(res_metrics_np)
    
    return metrics      

def train_flax(args):
    rounds = args.rounds
    batch_size = args.batch_size
    dataset = args.dataset
    network = args.network

    rng = jax.random.PRNGKey(0)

    # Make the dataset.
    rng, dataset_rng = jax.random.split(rng)
    (train_loader, test_loader), _, n_targets, (dummy_input, dummy_targets) = get_dataset_by_name(
        dataset, batch_size, batch_size, k_batches=args.k_batches, rng=dataset_rng, maxn=args.max_batch)

    net = get_flax_network(network, n_targets)
    create_train_state, train_step, eval_step, combine_train = flax_get_train_methods(net, dummy_input)

    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, args.learning_rate)

    if args.defense is not None:
        rng, defense_rng = jax.random.split(rng)
        dummy_grad = jax.grad(lambda p: flax_cross_entropy_loss(log_probs=net.apply({'params': p}, dummy_input), labels=dummy_targets))(state.params)
        _, def_perturb_grads, _, defense_params = get_defense(args.defense, defense_rng, args.batch_size, dummy_input, dummy_grad)
    else:
        def_perturb_grads, _, defense_params = None, None, None

    defend_grad, nodefend_grad = get_defend_grad(net, def_perturb_grads, args.batch_size, args.learning_rate, args.epochs, args.rand_batch)
    if args.defense is None:
        defend_grad = nodefend_grad
    
    def train_epoch(state, train_loader, batch_size, epoch, defense_params, rng):
        batch_metrics = []
        params = []
        for i in range(args.n_clients):
            inputs, targets = next(train_loader)
            rng, iter_rng = jax.random.split(rng)
            if args.dataset == 'FEMNIST' or args.dataset == 'CelebA':
                inputs = inputs[0]
                targets = targets[0]
            if len(inputs.shape) == 4:
                inputs = np.einsum('bijk -> bjki', inputs)
            log_probs = net.apply({'params': state.params}, inputs)
            grads, orders = defend_grad(iter_rng, state.params, defense_params, inputs, targets)
            n_state, metrics = train_step(state, grads, log_probs, targets, iter_rng)
            params.append(n_state.params)
            batch_metrics.append(metrics)
        params = jax.tree_multimap( lambda *args: jnp.stack(args), *params)
        params = combine_train(params)
        state = create_train_state(init_rng, args.learning_rate, params=params)

        epoch_metrics_np = aggregate_batch_metrics(batch_metrics)
        return state, epoch_metrics_np['loss'], epoch_metrics_np['accuracy'], epoch_metrics_np['grad_l2']

    def eval_model(params, test_loader):
        steps = args.train_steps if args.train_steps > 0 else None
        batch_metrics = []
        for inputs, targets in test_loader:
            if args.dataset == 'FEMNIST' or args.dataset == 'CelebA':
                inputs = inputs[0]
                targets = targets[0]
            if len(inputs.shape) == 4:
                inputs = np.einsum('bijk -> bjki', inputs)
            batch = {'image': inputs, 'label': targets}
            metrics = eval_step(params, batch)
            metrics = jax.device_get(metrics)
            batch_metrics.append(metrics)

        test_metrics_np = aggregate_batch_metrics(batch_metrics)
        return test_metrics_np['loss'], test_metrics_np['accuracy']


    tl = iter(train_loader)
    for epoch in range(rounds):
        try:
            state, train_loss, train_acc, grad_l2 = train_epoch(state, tl, batch_size, epoch, defense_params, rng)
            print('[train] round: %d, loss: %.4f, accuracy: %.4f, grad_l2: %.2f' % (epoch, train_loss, train_acc, grad_l2))
        except StopIteration:
            tl = iter(train_loader)
            state, train_loss, train_acc, grad_l2 = train_epoch(state, tl, batch_size, epoch, defense_params, rng)
            print('[train] round: %d, loss: %.4f, accuracy: %.4f, grad_l2: %.2f' % (epoch, train_loss, train_acc, grad_l2))
        if epoch % args.save_every == 0:
            test_loss, test_acc = eval_model(state.params, test_loader)
            print('[test] loss: %.4f, accuracy: %.4f' % (test_loss, test_acc))
            name = f"{args.dataset}_{args.network}_rounds_{epoch}_time_{datetime.now().strftime('%H-%M-%S')}_flax.pickle"
            path = os.path.join(*[args.prefix, name])
            store_model_flax(path, state)


    test_loss, test_acc = eval_model(state.params, test_loader)
    print('[test] loss: %.4f, accuracy: %.4f' % (test_loss, test_acc))

    name = f"{args.dataset}_{args.network}_rounds_{args.rounds}_time_{datetime.now().strftime('%H-%M-%S')}_flax.pickle"
    path = os.path.join(*[args.prefix, name])
    store_model_flax(path, state)


if __name__ == "__main__":
    args = get_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)


    assert not (args.attack and args.defend), "Cannot attack and defend at the same time"
    if args.debug:
        jaxconfig.update('jax_disable_jit', True)

    if args.attack:
        print(attack_flax(args, client_id=0))
    else:
        train_flax(args)


