import jax
import jax.numpy as jnp
import jax.scipy.optimize as jso
import numpy as np
import time
import torch
from tqdm import trange
from utils.flax_losses import flax_cross_entropy_loss, flax_compute_metrics, flax_get_attack_loss_and_update
from utils.util import generate_init_img
from utils.measures import l2_distance, compute_noise_l2
from utils.plotting import visualize
from scipy.optimize import linear_sum_assignment, minimize
from scipy.spatial import distance_matrix
from flax import linen as nn
from skimage.metrics import peak_signal_noise_ratio
from torchvision import transforms
from datasets.common import dataset_cfg


def compute_matching(at_inputs, inputs, delta, dataset, batch_size, metric='l2', fedavg=True):
    if (metric == 'mse' or metric == 'psnr') and (not dataset.startswith('dist_gaussian')):
        at_inputs = np.array(at_inputs)
        inputs = np.array(inputs)
        inv_normalize = transforms.Normalize(mean=dataset_cfg[dataset]['inv_mean'], std=dataset_cfg[dataset]['inv_std'])
        at_inputs = np.einsum("bijk -> bkij", at_inputs)
        inputs = np.einsum("bijk -> bkij", inputs)
        for i in range(inputs.shape[0]):
            at_inputs[i] = inv_normalize(torch.Tensor(np.array(at_inputs[i]))).cpu().detach().numpy()
            inputs[i] = inv_normalize(torch.Tensor(np.array(inputs[i]))).cpu().detach().numpy()

    #k_batches = (inputs.shape[0] + batch_size - 1) // batch_size
    
    #if fedavg == '':
    batch_size = inputs.shape[0]
    k_batches = 1

    all_diff, all_above_delta = [], []
    for idx in range(k_batches):
        st_idx = idx*batch_size
        en_idx = min( (idx+1)*batch_size, inputs.shape[0] )
        at_batch_inputs = at_inputs[st_idx:en_idx]
        batch_inputs = inputs[st_idx:en_idx]
        cost = np.zeros((batch_size, batch_size))
        for i in range(en_idx - st_idx):
            for j in range(en_idx - st_idx):
                if metric == 'l2':
                    cost[i, j] = np.sqrt(((at_batch_inputs[i] - batch_inputs[j])**2).sum())
                elif metric == 'mse':
                    cost[i, j] = ((at_batch_inputs[i] - batch_inputs[j])**2).sum() / at_batch_inputs[i].size
                elif metric == 'psnr':
                    cost[i, j] = -peak_signal_noise_ratio(at_batch_inputs[i], batch_inputs[j], data_range=1) # ???
                else:
                    assert False

        row_ind, col_ind = linear_sum_assignment(cost)
        if metric == 'psnr':
            cost = -cost
        diff = cost[row_ind, col_ind].mean()
        above_delta = (cost[row_ind, col_ind] > delta).mean()
        all_diff += [diff]
        all_above_delta += [above_delta]
        
        # Get order works only for single batch 
        order = np.argsort(row_ind)
        order = col_ind[order]
        assert k_batches == 1
    return np.mean(all_diff), np.mean(all_above_delta), order

def get_att_losses(net, prior, args):
    at_opt, _, _, at_update, restore_labels = flax_get_attack_loss_and_update(
            net, prior, args.optimizer, args.att_lr, args.batch_size, args=args)
    return at_opt, at_update, restore_labels

def attack_via_grad_opt_flax( root_dir,  compiled_att_funcs, rng, net, defend_grad, defense_params, net_params, inputs, targets, n_targets, prior, args=None):
    at_opt, at_update, restore_labels = compiled_att_funcs
    rng, defense_rng = jax.random.split(rng)
    noisy_grads, orders = defend_grad(defense_rng, net_params, defense_params, inputs, targets)

    start = args.att_fac_start
    alpha = np.exp(1.0/(args.att_epochs) * np.log(args.att_total_var/start))
    minmax = args.att_total_var

    curr_fac = start
    
    inv_mean = np.array( dataset_cfg[args.dataset]['inv_mean'] )
    inv_std = np.array( dataset_cfg[args.dataset]['inv_std'] )

    rng, defense_rng = jax.random.split(rng)
    rand_conv = nn.Conv(features=96, kernel_size=(3, 3))
    rand_conv_par = rand_conv.init(rng, inputs)
    
    tar_err = 0     
    if not args.restore_label is None: 
        rng, restore_rng = jax.random.split(rng)
        restored_targets = restore_labels(restore_rng, net_params, inputs, noisy_grads) 
        gt = [ np.sum(targets==i) for i in range(n_targets) ]
        tar_err = jnp.sum( jnp.abs(jnp.array(gt) - restored_targets) ) / 2
        l = [ [i]*x for i,x in enumerate(restored_targets)]
        targets = jnp.array( [item for sublist in l for item in sublist] )

    init_shape = inputs.shape

    if not args.rand_batch and not args.fedavg.endswith('known_labels'):
        rand_order = np.random.permutation(targets.shape[0])
        targets = targets[ rand_order ]

    targets = np.tile( targets, args.epochs )
    if 'many' in args.fedavg:
        init_shape = [args.epochs*init_shape[0]] + list( init_shape[1:] )
    if args.fedavg.endswith('known_labels'):
        targets = targets[ orders.reshape(-1) ]
 
    if args.restore_label is None or not 'only' in args.restore_label:
        max_psnr = None
        best_image = None
        for idx in range(args.att_restarts):
            rng, init_rng = jax.random.split(rng)
            at_img = generate_init_img(init_rng, init_shape, args.att_init)
            at_opt_state = at_opt.init(at_img)

            if args.visualize:
                visualize(inputs, f'{root_dir}/reference.png', source=args.dataset, batch_size=args.batch_size)
                visualize(at_img, f"{root_dir}/at_img_0.png", source=args.dataset, batch_size=args.batch_size)

            curr_fac = start

            for at_iter in range(2*args.att_epochs):
                start_time = time.time()
                curr_fac = np.minimum( curr_fac * alpha, minmax )  if alpha >= 1.0 else np.maximum( curr_fac * alpha, minmax )
                
                at_img_new, at_opt_state_new, att_loss_new = at_update(net_params, at_opt_state, at_img, targets, noisy_grads, curr_fac, orders, rand_conv_par, inv_mean, inv_std)
                
                if jnp.any(jnp.isnan(at_img_new)).item(): # If diverged, happens rarely with very bad settings 
                    break
                at_img, at_opt_state, att_loss = at_img_new, at_opt_state_new, att_loss_new

                end_time = time.time()
                if at_iter % args.vis_step == 0:
                    visualize(at_img, f"{root_dir}/at_img_{at_iter+1}.png", source=args.dataset, batch_size=args.batch_size)

            rec_orders = []
            if 'many' in args.fedavg:
                at_img = at_img.reshape(-1,*inputs.shape)
                at_img_reorder = [at_img[0]]
                for j in range(1,at_img.shape[0]):
                    _, _, opt_order = compute_matching(at_img[0], at_img[j], args.delta, args.dataset, args.batch_size, metric='psnr', fedavg=args.fedavg)
                    at_img_reorder.append( at_img[j][opt_order] )
                    rec_orders.append( opt_order )
                rec_orders = jnp.concatenate( rec_orders )
                at_img = jnp.stack( at_img_reorder )
                at_img = at_img.mean(axis=0)

            diff, _, _ = compute_matching(at_img, inputs, args.delta, args.dataset, args.batch_size, fedavg=args.fedavg)
            diff_mse, _, _ = compute_matching(at_img, inputs, args.delta, args.dataset, args.batch_size, metric='mse', fedavg=args.fedavg)
            diff_psnr, above_delta, opt_order = compute_matching(at_img, inputs, args.delta, args.dataset, args.batch_size, metric='psnr', fedavg=args.fedavg)
            if max_psnr is None or diff_psnr > max_psnr:
                max_psnr = diff_psnr
                opt_order = jnp.argsort(opt_order)
                best_image = at_img[opt_order]
                best_diff = diff
                best_mse = diff_mse
                best_above_delta = above_delta
                best_order = rec_orders
        
        metrics = {'diff': best_diff, 'diff_mse': best_mse, 'diff_psnr': max_psnr, 'above_delta': best_above_delta, 'target_errors':tar_err, 'best_order': str(best_order) }
        if args.visualize:
            visualize(best_image, f"{root_dir}/at_img_{at_iter+1}.png", source=args.dataset, batch_size=args.batch_size)
    else:
        metrics = {'target_errors':tar_err}

    return metrics
