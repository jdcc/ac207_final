from functools import partial

from jax import random, value_and_grad, jit
from jax.experimental.optimizers import adam
import jax.numpy as jnp
import jax.scipy as jsp
from tqdm.auto import trange

def unpack_latent_params(z):
    z_dim = z.shape[1] // 2
    mean = z[:,:z_dim]
    std = jnp.exp(z[:,z_dim:])
    return mean, std

def log_likelihood_data(rng_key, model, params, data, data_vari, n_latent_samples):
    '''Expected log-likelihood of data under current model'''
    encode, decode = model
    enc_params, dec_params = params
    latent_params = encode(enc_params, data)
    mean, std = unpack_latent_params(latent_params)
    latent_dim = mean.shape[1]
    n_obs = data.shape[0]
    latent_samples = random.normal(rng_key, shape=(n_latent_samples, n_obs, latent_dim)) * std + mean
    reconstructed_data = decode(dec_params, latent_samples)
    log_likelihood = jsp.stats.norm.logpdf(data, reconstructed_data, data_vari**0.5).sum(axis=-1)
    assert log_likelihood.shape == (n_latent_samples, n_obs)
    return log_likelihood, latent_samples, mean, std

def log_prob_latent_under_prior(latent_samples):
    ''' Log-probability of latent vars under N(0,1) prior '''
    return jsp.stats.norm.logpdf(latent_samples, 0, 1).sum(axis=-1)

def log_prob_latent_under_variational(latent_samples, mean, std):
    ''' Log-probability of latent vars under variational dist '''
    return jsp.stats.norm.logpdf(latent_samples, mean, std).sum(axis=-1)
    
def general_objective(params, rng_key, model, data, data_vari, n_latent_samples=50):
    log_likelihood, latent_samples, mean, std = log_likelihood_data(
        rng_key, model, params, data, data_vari, n_latent_samples)
    log_prior_prob_latent = log_prob_latent_under_prior(latent_samples)
    log_variational_prob_latent = log_prob_latent_under_variational(latent_samples, mean, std)
    
    # Estimated expected value
    elbo = (log_likelihood - log_variational_prob_latent + log_prior_prob_latent).mean()
    return -elbo

def latent_dim_from_params(params):
    # Locate the number of output layer biases
    return len(params[-1][-2][-1])

def infer(model, params, data):
    encode, _ = model
    enc_params, _ = params
    return encode(enc_params, data)

def generate_samples(model, params, rng_key, n=100):
    latent_dim = latent_dim_from_params(params)
    latent_samples = random.normal(rng_key, (n, latent_dim))
    _, decode = model
    _, dec_params = params
    return decode(dec_params, latent_samples)

def fit(rng_key, model, start_params, data, data_vari, step_size=1e-3, max_iter=1000):
    '''
    Args:
      *model: tuple of (encode, decode)
      *start_params: weights and biases
      *data: array like (obs, features)
    '''
    opt_init, update_params, get_params = adam(step_size)
    opt_state = opt_init(start_params)
    history = []
    my_elbo = jit(partial(general_objective, model=model, data=data, data_vari=data_vari))
    min_loss_params = (1e10, None)
    for i in trange(max_iter, smoothing=0):
        params = get_params(opt_state)
        rng_key, subkey = random.split(rng_key)
        loss, grads = value_and_grad(my_elbo)(params, subkey)
        opt_state = update_params(i, grads, opt_state)
        if loss < min_loss_params[0]:
            min_loss_params = (loss, params)
        history.append((loss,))
    return min_loss_params, history