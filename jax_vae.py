from jax_nn import create_mlp, mse_loss, binary_crossentropy_loss
from jax.experimental.stax import Relu, Sigmoid, Identity

def objective(params, data, encode, decode):
    z_params = encode(data)
    
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
            self.random_key, subkey = random.split(self.random_key)
            z_samples = random.normal(subkey, shape=(self.S, self.z_dim, N)) * std + mean
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
            
        return variational_objective