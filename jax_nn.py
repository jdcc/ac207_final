from functools import partial

from jax import value_and_grad, random, jit, device_put
from jax.experimental import stax
from jax.experimental.optimizers import adam
import jax.scipy as jsp
import jax.numpy as jnp
from tqdm.auto import trange

def create_mlp(hidden_widths, activation_fn, output_dim, output_activation_fn=stax.Identity):
    layers = []
    for width in hidden_widths:
        layers.extend([stax.Dense(width), activation_fn])
    layers.extend([stax.Dense(output_dim), output_activation_fn])
    return stax.serial(*layers)

def binary_crossentropy_loss(params, predict, data):
    inputs, targets = data
    probs = predict(params, inputs)
    eps = jnp.finfo(probs.dtype).eps
    probs = jnp.clip(probs, eps, 1 - eps)
    loss = -(jsp.special.xlogy(targets, probs) + jsp.special.xlogy(1 - targets, 1 - probs)).mean()
    return loss

def mse_loss(params, predict, data):
    inputs, targets = data
    preds = predict(params, inputs)
    loss = ((preds - targets)**2).mean()
    return loss

def fit(model, start_params, data, step_size=1e-3, max_iter=1000):
    '''
    Args:
      *model: tuple of (predict, calc_loss)
      *start_params: weights and biases
      *data: data like (X, y)
    '''
    opt_init, update_params, get_params = adam(step_size)
    opt_state = opt_init(start_params)
    history = []
    step_model = jit(partial(fit_step, model, data=data))
    for i in trange(max_iter, smoothing=0):
        params = get_params(opt_state)
        loss, grads = step_model(params)
        opt_state = update_params(i, grads, opt_state)
        output_layer_weights = grads[-2][0]
        output_layer_weight_mag = ((output_layer_weights.T@output_layer_weights)**0.5)[0][0]
        history.append((loss, output_layer_weight_mag))
    return get_params(opt_state), history

def fit_step(model, params, data):
    predict, calc_loss = model
    loss, grads = value_and_grad(calc_loss)(params, predict, data)
    return loss, grads