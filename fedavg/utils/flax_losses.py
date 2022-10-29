import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training import train_state
from utils.measures import total_variation
from utils.util import ball_get_random
from datasets.distributions import VAE_Prior, GMM_Prior

@jax.jit
def flax_cross_entropy_loss(*, log_probs, labels):
    one_hot_labels = jax.nn.one_hot(labels, num_classes=log_probs.shape[1])
    return -jnp.mean(jnp.sum(one_hot_labels * log_probs, axis=-1))


@jax.jit
def flax_compute_metrics(*, log_probs, labels):
    loss = flax_cross_entropy_loss(log_probs=log_probs, labels=labels)
    accuracy = jnp.mean(jnp.argmax(log_probs, -1) == labels)
    metrics = {'loss': loss, 'accuracy': accuracy}
    return metrics


@jax.jit
def cos_sim(x, y):
    x_l2_sqr = sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(x))
    y_l2_sqr = sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(y))
    return 1 - jnp.sum(x * y) / (jnp.sqrt(x_l2_sqr + 1e-7) * jnp.sqrt(y_l2_sqr + 1e-7))


@jax.jit
def l2_dist(x, y):
    x_l2_sqr = sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(x))
    y_l2_sqr = sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(y))
    return jnp.sum(jnp.square(x/jnp.sqrt(x_l2_sqr+1e-7) - y/jnp.sqrt(y_l2_sqr+1e-7)))

@jax.jit
def clip_prior(x, mean, std ):
    x_unorm = (x - mean)/ std  
    dist_clip = jnp.sum( jnp.mean( jnp.square(  x_unorm - jnp.clip( x_unorm, 0.0, 1.0) ), axis=0) )
    return dist_clip

@jax.jit
def invariant_prior_l2_mean(idx1, idx2, inputs):
    x1 = inputs[idx1]
    x2 = inputs[idx2]
    x1 = x1.mean(axis=0) 
    x2 = x2.mean(axis=0)
    error = jnp.mean(jnp.square(x2-x1))
    return error

@jax.jit
def invariant_prior_l2_max(idx1, idx2, inputs):
    x1 = inputs[idx1]
    x2 = inputs[idx2]
    x1 = x1.max(axis=0) 
    x2 = x2.max(axis=0)
    error = jnp.mean(jnp.square(x2-x1))
    return error

@jax.jit
def invariant_prior_l1_mean(idx1, idx2, inputs):
    x1 = inputs[idx1]
    x2 = inputs[idx2]
    x1 = x1.mean(axis=0) 
    x2 = x2.mean(axis=0)
    error = jnp.mean(jnp.abs(x2-x1))
    return error

@jax.jit
def invariant_prior_l1_max(idx1, idx2, inputs):
    x1 = inputs[idx1]
    x2 = inputs[idx2]
    x1 = x1.max(axis=0) 
    x2 = x2.max(axis=0)
    error = jnp.mean(jnp.abs(x2-x1))
    return error


def flax_get_train_methods(net, dummy_input):
    def create_train_state(rng, learning_rate, momentum=None, params=None, opt='sgd'):
        if params is None:
            params = net.init(rng, dummy_input)['params']
        if opt == 'adam':
            tx = optax.adam(learning_rate)
        elif opt == 'sgd':
            tx = optax.sgd(learning_rate, momentum)
        return train_state.TrainState.create(apply_fn=net.apply, params=params, tx=tx)

    @jax.jit
    def train_step(state, grads, log_probs, labels, rng=None):
        state = state.apply_gradients(grads=grads)
        metrics = flax_compute_metrics(log_probs=log_probs, labels=labels)
        metrics['grad_l2'] = sum(jax.tree_leaves(jax.tree_map(lambda x: (x**2).sum(), grads)))
        return state, metrics
    
    @jax.jit
    def combine_train(params):
        avg_params = jax.tree_map(lambda x: jnp.mean(x, axis=0), params)
        return avg_params

    @jax.jit
    def eval_step(params, batch):
        log_probs = net.apply({'params': params}, batch['image'])
        return flax_compute_metrics(log_probs=log_probs, labels=batch['label'])

    return create_train_state, train_step, eval_step, combine_train

# Attack loss functions
def flax_get_attack_loss_and_update(net, prior, opt_str, learning_rate, batch_size, args, is_train=False):
    exp_schedule = optax.exponential_decay(learning_rate, args.exp_decay_steps, args.exp_decay_factor)  # TODO: Tune this
    opt = optax.adam(learning_rate=exp_schedule)

    @jax.jit
    def get_orig_grads(net_params, net_grad, inputs, targets, orders):

        @jax.jit
        def single_grad(input, target, params):
            inputs, targets = jnp.expand_dims(input, axis=0), jnp.array([target])
            return jax.grad(lambda p: flax_cross_entropy_loss(log_probs=net.apply({'params': p}, inputs), labels=targets))(params)
        
        @jax.jit
        def interpolate_weights(st_params, net_grad, coef, lr): 
            return jax.tree_multimap(lambda x, g: x - coef*lr*g, st_params, net_grad)

        @jax.jit
        def batch_grad( inputs, targets, params ):
            grads = jax.vmap(single_grad, (0, 0, None))(inputs, targets, params)
            grad_avg = jax.tree_map(lambda x: jnp.mean(x, axis=0), grads)
            #print(  'Grad:', jax.tree_map(lambda x: x.shape, grads), 'Ins:', inputs.shape, 'Params:', jax.tree_map(lambda x: x.shape, params) )
            return grad_avg

        @jax.jit
        def batch_grad_many_par( inputs, targets, params ):
            grads = jax.vmap(single_grad, (0, 0, 0))(inputs, targets, params)
            grad_avg = jax.tree_map(lambda x: jnp.mean(x, axis=0), grads)
            #print(  'Grad:', jax.tree_map(lambda x: x.shape, grads), 'Ins:', inputs.shape, 'Params:', jax.tree_map(lambda x: x.shape, params) )
            return grad_avg

        epoch_size = inputs.shape[0]
        if 'many' in args.fedavg:
            epoch_size = inputs.shape[0] // args.epochs  
        k_batches = (epoch_size + batch_size-1) // batch_size

        if args.fedavg == 'full_known_labels':
            params = net_params
            real_epoch_size = targets.shape[0] // args.epochs
            for j in range(args.epochs):
                for i in range(k_batches):
                    st_idx = i*batch_size
                    en_idx = min( (i+1)*batch_size, inputs.shape[0] )
                    grad_avg = batch_grad(inputs[orders[j]][st_idx:en_idx], targets[st_idx + j * real_epoch_size:en_idx + j * real_epoch_size], params)
                    params = jax.tree_multimap(lambda p, g:  p - args.learning_rate*g, params, grad_avg)
            return jax.tree_multimap(lambda x_f, x_s: (x_f - x_s)/(-args.learning_rate), params, net_params)
        if args.fedavg.startswith('full_many'):
            params = net_params
            for j in range(args.epochs):
                for i in range(k_batches):
                    st_idx = i*batch_size + j * epoch_size
                    en_idx = min( (i+1)*batch_size, inputs.shape[0] ) + j * epoch_size
                    grad_avg = batch_grad(inputs[st_idx:en_idx], targets[st_idx:en_idx], params)
                    params = jax.tree_multimap(lambda p, g:  p - args.learning_rate*g, params, grad_avg)
            return jax.tree_multimap(lambda x_f, x_s: (x_f - x_s)/(-args.learning_rate), params, net_params)
        if args.fedavg == 'full':
            params = net_params
            for j in range(args.epochs):
                for i in range(k_batches):
                    st_idx = i*batch_size
                    en_idx = min( (i+1)*batch_size, inputs.shape[0] )
                    grad_avg = batch_grad(inputs[st_idx:en_idx], targets[st_idx:en_idx], params)
                    params = jax.tree_multimap(lambda p, g:  p - args.learning_rate*g, params, grad_avg)
            return jax.tree_multimap(lambda x_f, x_s: (x_f - x_s)/(-args.learning_rate), params, net_params)
        elif args.fedavg == 'none':
            grad_avg = batch_grad(inputs, targets[:inputs.shape[0]], net_params)
            grad_avg = jax.tree_map(lambda x: x*k_batches*args.epochs, grad_avg)
            return grad_avg
        elif args.fedavg == 'none_epoch':
            params = net_params
            for j in range(args.epochs):
                grad_avg = batch_grad(inputs, targets[:inputs.shape[0]], params)
                grad_avg = jax.tree_map(lambda x: x*k_batches, grad_avg)
                params = jax.tree_multimap(lambda p, g:  p - args.learning_rate*g, params, grad_avg)
            return jax.tree_multimap(lambda x_f, x_s: (x_f - x_s)/(-args.learning_rate), params, net_params)
        elif args.fedavg == 'none_epoch_many':
            params = net_params
            for j in range(args.epochs):
                st_idx = j * epoch_size
                en_idx = (j+1) * epoch_size
                grad_avg = batch_grad(inputs[st_idx:en_idx], targets[st_idx:en_idx], params)
                grad_avg = jax.tree_map(lambda x: x*k_batches, grad_avg)
                params = jax.tree_multimap(lambda p, g:  p - args.learning_rate*g, params, grad_avg)
            return jax.tree_multimap(lambda x_f, x_s: (x_f - x_s)/(-args.learning_rate), params, net_params)
        else:
            assert False
    
    @jax.jit
    def at_internal(params, grads, inputs, targets, fac, orders, rand_conv_par, inv_mean=0.0, inv_std=1.0):
        k_batches = (inputs.shape[0] + batch_size-1) // batch_size
        att_grads = get_orig_grads(params, grads, inputs, targets, orders)

        tot_var = args.attack_total_variation * total_variation(inputs).mean()
        
        inv_prior = 0.0
        if not args.reorder_prior == 'none':
            epoch_size = inputs.shape[0] // args.epochs
            inv_prior = 0.0
            x = jnp.arange(args.epochs)
            y = jnp.arange(args.epochs)
            xv, yv = jnp.meshgrid(x, y)
            xv, yv = xv.reshape(-1), yv.reshape(-1)
            
            yv = jnp.tile(yv, epoch_size).reshape(epoch_size,-1).T*epoch_size + jnp.arange(epoch_size)
            xv = jnp.tile(xv, epoch_size).reshape(epoch_size,-1).T*epoch_size + jnp.arange(epoch_size)
            
            inputs_proj = inputs
            if args.reorder_prior.endswith('conv'):
                rand_conv = nn.Conv(features=96, kernel_size=(3, 3))
                inputs_proj = rand_conv.apply(rand_conv_par, inputs)

            if args.reorder_prior.startswith('l2_mean'):
                inv_prior = jax.vmap(invariant_prior_l2_mean, (0, 0, None))(xv, yv, inputs_proj)
            elif  args.reorder_prior.startswith('l2_max'):
                inv_prior = jax.vmap(invariant_prior_l2_max, (0, 0, None))(xv, yv, inputs_proj)
            elif  args.reorder_prior.startswith('l1_mean'):
                inv_prior = jax.vmap(invariant_prior_l1_mean, (0, 0, None))(xv, yv, inputs_proj)
            elif  args.reorder_prior.startswith('l1_max'):
                inv_prior = jax.vmap(invariant_prior_l1_max, (0, 0, None))(xv, yv, inputs_proj)
            else:
                assert False
            inv_prior = inv_prior.mean()

        layer_weights = np.arange(len(grads), 0, -1)
        layer_weights = np.exp( layer_weights )
        layer_weights = layer_weights / np.sum( layer_weights )
        layer_weights = layer_weights / layer_weights[0]
        layer_weights = np.repeat(layer_weights, 2)
        if not args.att_exp_layers:
            layer_weights = np.repeat([1.0], 2*len(grads))
        trs = jax.tree_structure( grads )
        layer_weights = jax.tree_unflatten(trs, layer_weights)
        weights = np.repeat([1.0], 2*len(grads))
        weights = jax.tree_unflatten(trs, weights)

        if args.att_metric == 'l2':
            res = jax.tree_multimap(lambda x, y, w, lw:  lw*jnp.sum(jnp.multiply(jnp.square(x-y),w)), grads, att_grads, weights, layer_weights)
            l2_loss = sum(p for p in jax.tree_leaves(res))
            att_loss = l2_loss
        elif args.att_metric == 'l1':
            res = jax.tree_multimap(lambda x, y, w, lw:  lw*jnp.sum(jnp.multiply(jnp.abs(x-y),w)), grads, att_grads, weights, layer_weights)
            l1_loss = sum(p for p in jax.tree_leaves(res))
            att_loss = l1_loss
        elif args.att_metric == 'cos_sim':
            res = jax.tree_multimap(lambda x, y, w, lw: lw*cos_sim(jnp.multiply(x,w), jnp.multiply(y,w)), grads, att_grads, weights, layer_weights)
            att_loss = sum(p for p in jax.tree_leaves(res))
        elif args.att_metric == 'cos_sim_global':
            dot = jax.tree_multimap(lambda x, y, w, lw: -lw*jnp.sum( jnp.multiply( jnp.multiply(x,w), jnp.multiply(y,w) ) ), grads, att_grads, weights, layer_weights)
            dot = sum(p for p in jax.tree_leaves(dot))
            norm1 = jax.tree_multimap(lambda x,w: jnp.sum( jnp.multiply(jnp.multiply(x,w),jnp.multiply(x,w)) ), grads, weights)
            norm1 = sum(p for p in jax.tree_leaves(norm1))
            norm2 = jax.tree_multimap(lambda x,w: jnp.sum(  jnp.multiply(jnp.multiply(x,w),jnp.multiply(x,w)) ), att_grads, weights)
            norm2 = sum(p for p in jax.tree_leaves(norm2))
            att_loss = 1 + dot / (jnp.sqrt(norm1 + 1e-7) * jnp.sqrt(norm2 + 1e-7))
        else:
            assert False
        clip_err = clip_prior(inputs, inv_mean, inv_std )
        att_loss /= k_batches
        #print( "rec: " + str((att_loss * fac).val) + ' totvar: ' + str( (args.reg_tv*tot_var).val ) + ' clip_err: ' + str( (args.reg_clip*clip_err).val ))
        tot_loss = att_loss * fac + args.reg_tv * tot_var + args.reg_clip * clip_err + args.reg_reorder * inv_prior

        return tot_loss

    def at_internal_region(rng, params, defense_params, grads, inputs, targets, fac, n_samples):
        @jax.jit
        def compute_one_sample(sample_rng):
            d = jnp.zeros(inputs.shape)
            return at_internal(rng, params, defense_params, grads, inputs + d, targets, fac)
        rngs = jax.random.split(rng, n_samples + 1)
        rng, sample_rngs = rngs[0], rngs[1:]
        losses = jax.vmap(compute_one_sample)(sample_rngs)
        return losses.mean()
    
    @jax.jit
    def at_update(params, opt_state, inputs, targets, grads, fac, orders, rand_conv_par, inv_mean, inv_std):
        att_grad_fn = jax.value_and_grad(at_internal, 2)
        att_loss, att_grad = att_grad_fn(params, grads, inputs, targets, fac, orders, rand_conv_par, inv_mean, inv_std)
        updates, opt_state = opt.update(att_grad, opt_state)
        new_att_value = optax.apply_updates(inputs, updates)
        return new_att_value, opt_state, att_loss

    @jax.jit
    def calc_AAAI_label_stats(rng, params, inputs, grads):
        @jax.jit
        def single_invoke(dummy_in):
            out, state = net.apply({'params': params}, dummy_in[None,], mutable=['label_pred'])
            last_relu = state['label_pred']['last_relu']
            return jnp.exp(out), last_relu

        # Run dummy data and extract softmax and last ReLU
        shape = inputs.shape 
        shape = (args.restore_label_samp, *shape[1:])
        dummy_in = jax.random.normal(rng, shape)
        outs = jax.vmap(single_invoke, (0))(dummy_in)
        ps, Os = outs[0], outs[1][0]
    
        # Compute means of ps and O, compute sum of derivatives on last layer
        ps = jnp.mean( ps, axis=[0,1] ) 
        O = jnp.mean( jnp.sum(Os, axis=-1) )
        dW = np.sum(grads['last_layer']['kernel'],axis=0)
        return dW, O, ps

    @jax.jit
    def calc_AAAI_raw_cnts(ps, O, dW, s, K):
        return K*ps - (K * dW) / O / s

    def round_label_cnts(cnts, K): 
        # Rounding can cause more than input.shape[0] labels. Prevent by taking max
        cnts_fl = jnp.maximum( jnp.array(jnp.floor( cnts ), int), jnp.array(0, int) )
        cnts_rem = K - jnp.sum( cnts_fl )
        cnts_rem_arr = cnts - cnts_fl
        if cnts_rem >= 0:
            _, idx = jax.lax.top_k( cnts_rem_arr, cnts_rem )
            cnts = cnts_fl.at[idx].set( cnts_fl[idx] + 1 )
        else:
            max_rm = -cnts_rem
            rem = 0
            if -cnts_rem > jnp.sum(cnts_fl >= 0.1 ):
                rem = -cnts_rem - jnp.sum(cnts_fl >= 0.1 )
                max_rm = jnp.sum(cnts_fl >= 0.1 )

            cnts_rem_arr = cnts_rem_arr.at[ cnts_fl <= 0.1 ].set(1)
            _, idx = jax.lax.top_k( -cnts_rem_arr, max_rm )
            cnts = cnts_fl.at[idx].set( cnts_fl[idx] - 1 )
            if rem > 0:
                _, idx = jax.lax.top_k( cnts_fl, rem )
                cnts = cnts.at[idx].set( cnts[idx] - 1 )
        return cnts

    def restore_labels(rng, server_params, inputs, grads):
        K = inputs.shape[0]
        k_batches = (inputs.shape[0] + batch_size-1) // batch_size

        client_params = jax.tree_multimap(lambda p, g:  p - args.learning_rate*g, server_params, grads)
        dW, O_start, ps_start = calc_AAAI_label_stats(rng, server_params, inputs, grads)
        _, O_end, ps_end = calc_AAAI_label_stats(rng, client_params, inputs, grads)

        coefs = jnp.arange( 0, 1, 1/k_batches/args.epochs )
        Os = (1 - coefs) * O_start + coefs * O_end
        coefs = coefs.reshape(-1,1)
        ps_start, ps_end = ps_start.reshape(1, -1), ps_end.reshape(1, -1)
        ps = (1 - coefs) @ ps_start + coefs @ ps_end
        
        if args.restore_label.startswith('aaai_st'):
            idx_st = 0
            idx_en = 1
        elif args.restore_label.startswith('aaai_en'):
            idx_st = -1
            idx_en = 0
        elif args.restore_label.startswith('aaai_avg'):
            idx_st = 0
            idx_en = k_batches * args.epochs
        else:
            assert False

        raw_cnts = []
        for j in range( idx_st, idx_en ):    
            cnts = calc_AAAI_raw_cnts(ps[j], Os[j], dW, k_batches * args.epochs, K)
            raw_cnts.append( cnts )
        raw_cnts = jnp.stack(raw_cnts)
        cnts = round_label_cnts(raw_cnts.mean(axis=0), K)
        return cnts

    jit_at_internal_region = jax.jit(at_internal_region, static_argnums=(7,))

    return opt, at_internal, jit_at_internal_region, at_update, restore_labels 
