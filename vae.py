from jax import jit, value_and_grad
from jax.experimental.optimizers import adam
import jax.numpy as np
import jax.scipy.stats.norm as norm

import numpy # This is used for randomness, but should be replaced

from nn_models import Feedforward

class VAE:
    def __init__(self, decoder_architecture, encoder_architecture, x_var, random=None, decoder_weights=None, encoder_weights=None):
        '''constructor'''
        self.x_dim = decoder_architecture['output_dim']
        self.z_dim = decoder_architecture['input_dim']

        self.x_var = x_var

        assert encoder_architecture['input_dim'] == self.x_dim
        assert encoder_architecture['output_dim'] == self.z_dim * 2

        self.decoder = Feedforward(decoder_architecture, random=random, weights=decoder_weights)
        self.encoder = Feedforward(encoder_architecture, random=random, weights=encoder_weights)

        self.objective_trace = np.empty((1, 1))
        self.param_trace = np.empty((1, self.decoder.D + self.encoder.D))
        self.S = 10

        if random is not None:
            self.random = random
        else:
            self.random = numpy.random.RandomState(0)
            
    def generate(self, weights=None, N=100):
        '''use the generative model to generate x given zs sampled from the prior'''
        z_samples = self.random.normal(0, 1, size=(self.z_dim, N))

        if weights is None:
            x_samples = self.decoder.forward(self.decoder.weights, z_samples)
        else:
            x_samples = self.decoder.forward(weights, z_samples)
            
        return x_samples[0]

    def infer(self, x, weights=None):
        '''use the inference model to infer parameters of q(z|x)'''
        if weights is None:
            z = self.encoder.forward(self.encoder.weights, x)
        else:
            z = self.encoder.forward(weights, x)
            
        return z[0]
    
    def unpack_params(self, z_params):
        '''unpack variational parameters for q(z|x)'''
        assert len(z_params.shape) == 2
        assert z_params.shape[0] == 2 * self.z_dim

        mean, parametrized_var = z_params[:self.z_dim], z_params[self.z_dim:]
        std = np.exp(parametrized_var)

        return mean, std
    
    def unpack_weights(self, params):
        '''unpack the weights for the encoder and decoder'''
        assert len(params.shape) == 2
        assert params.shape[1] == self.encoder.D + self.decoder.D
        
        decoder_weights = params[:, :self.decoder.D].reshape((1, -1))
        encoder_weights = params[:, self.decoder.D:].reshape((1, -1))
        
        return encoder_weights, decoder_weights
    
    def make_objective(self, x_train, S):
        '''make variational objective function and gradient of the variational objective'''
        assert len(x_train.shape) == 2
        assert x_train.shape[0] == self.x_dim
        

        if S is not None:
            self.S = S

        N = x_train.shape[1]
        x_dummy = np.zeros((self.S, self.x_dim, N))

           
        def variational_objective(params):
            '''definition of the ELBO'''
            encoder_weights, decoder_weights = self.unpack_weights(params)

            #infer z's
            z_params = self.encoder.forward(encoder_weights, x_train)[0]
            
            #unpack var parameters
            mean, std = self.unpack_params(z_params)
            assert std.shape == (self.z_dim, N)
            assert mean.shape == (self.z_dim, N)
            
            #sample z's
            z_samples = numpy.random.normal(0, 1, size=(self.S, self.z_dim, N)) * std + mean
            assert z_samples.shape == (self.S, self.z_dim, N)
            
            #predict x's
            x = self.decoder.forward(decoder_weights, z_samples)
            assert x.shape == (self.S, self.x_dim, N)
            
            #evaluate log-likelihood
            log_likelihood = np.sum(norm.logpdf(x_train, x, self.x_var ** 0.5), axis=-2)
            assert log_likelihood.shape == (self.S, N)
                        
            #evaluate sampled z's under prior
            log_pz = np.sum(norm.logpdf(z_samples, 0.0, 1.0), axis=-2)
            assert log_pz.shape == (self.S, N)
            
            #evaluate sampled z's under variational distribution
            log_qz_given_x = np.sum(norm.logpdf(z_samples, mean, std), axis=-2)
            assert log_qz_given_x.shape == (self.S, N)
            
            #compute the elbo
            elbo = np.mean(log_likelihood - log_qz_given_x + log_pz)
            
            #return the negative elbo to be minimized
            return -elbo
            
            
        return jit(variational_objective)
    
                          
    def fit(self, x_train, S=None, params=None):
        '''minimize -ELBO'''
        assert x_train.shape[0] == self.x_dim

        ### make objective function for training
        objective = self.make_objective(x_train, S)

        ### set up optimization
        step_size = 0.01
        max_iteration = 5000
        check_point = 100
        param_init = self.random.normal(0, 0.3, size=(1, self.decoder.D + self.encoder.D))
        mass = None
        optimizer = 'adam'
        random_restarts = 1

        if 'step_size' in params.keys():
            step_size = params['step_size']
        if 'max_iteration' in params.keys():
            max_iteration = params['max_iteration']
        if 'check_point' in params.keys():
            check_point = params['check_point']
        if 'init' in params.keys():
            param_init = params['init']
        if 'call_back' in params.keys():
            call_back = params['call_back']
        if 'mass' in params.keys():
            mass = params['mass']
        if 'optimizer' in params.keys():
            optimizer = params['optimizer']
        if 'random_restarts' in params.keys():
            random_restarts = params['random_restarts']

        ### train with random restarts
        optimal_obj = 1e16
        optimal_param = param_init

        for i in range(random_restarts):
            if optimizer == 'adam':
                opt_init, opt_update, get_params = adam(step_size=step_size)
                opt_state = opt_init(param_init)
                
                def step(iteration, opt_state):
                    params = get_params(opt_state)
                    objective_val, grads = value_and_grad(objective)(params)
                    self.objective_trace = np.vstack((self.objective_trace, objective_val))
                    self.param_trace = np.vstack((self.param_trace, params))
                    opt_state = opt_update(iteration, grads, opt_state)
                    if iteration % check_point == 0:
                        print(f"Iteration {iteration} lower bound {objective_val:.4f}; gradient mag: {np.linalg.norm(grads):.4f}")
                    return objective_val, opt_state

                for i in range(max_iteration):
                    objective_val, opt_state = step(i, opt_state)
                
                #adam(gradient, param_init, step_size=step_size, num_iters=max_iteration, callback=call_back)
            local_opt = np.min(self.objective_trace[1:])
            
            if local_opt < optimal_obj:
                opt_index = np.argmin(self.objective_trace[1:])
                opt_params = self.param_trace[1:][opt_index].reshape((1, -1))
                self.opt_params = opt_params
                encoder_weights, decoder_weights = self.unpack_weights(opt_params)
                self.decoder.weights = decoder_weights
                self.encoder.weights = encoder_weights
                optimal_obj = local_opt

            param_init = self.random.normal(0, 0.1, size=(1, self.decoder.D + self.encoder.D))

        self.objective_trace = self.objective_trace[1:]
        self.param_trace = self.param_trace[1:]