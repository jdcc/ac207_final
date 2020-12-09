import jax.numpy as np
from jax import grad
from jax.experimental.optimizers import adam

class Feedforward:
    def __init__(self, architecture, random=None, weights=None):
        self.params = {'H': architecture['width'],
                       'L': architecture['hidden_layers'],
                       'D_in': architecture['input_dim'],
                       'D_out': architecture['output_dim'],
                       'activation_type': architecture['activation_fn_type'],
                       'activation_params': architecture['activation_fn_params'],
                       'output_activation_type': architecture['output_activation_type'],
                       'output_activation_fn': architecture['output_activation_fn'],
                       'task': architecture['task'],
                      }

        # Make sure we have a width for every hidden layer
        assert len(self.params['H']) == self.params['L']
        
        # Make sure we can pick a good 
        assert self.params['task'] in ['regression', 'classification']
        
        self.D = (  (architecture['input_dim'] * self.params['H'][0] + self.params['H'][0])
                  + (architecture['output_dim'] * self.params['H'][-1] + architecture['output_dim'])
                 )
        for i in range(len(self.params['H'])-1):
            self.D += self.params['H'][i] * self.params['H'][i+1] + self.params['H'][i+1]

        if random is not None:
            self.random = random
        else:
            self.random = np.random.RandomState(0)

        self.h = architecture['activation_fn']

        if weights is None:
            self.weights = self.random.normal(0, 1, size=(1, self.D))
        else:
            self.weights = weights

        self.objective_trace = np.empty((1, 1))
        self.weight_trace = np.empty((1, self.D))


    def forward(self, weights, x):
        ''' Forward pass given weights and input '''
        H = self.params['H']
        D_in = self.params['D_in']
        D_out = self.params['D_out']

        assert weights.shape[1] == self.D

        if len(x.shape) == 2:
            assert x.shape[0] == D_in
            x = x.reshape((1, D_in, -1))
        else:
            assert x.shape[1] == D_in

        weights = weights.T


        #input to first hidden layer
        first_H = H[0]
        W = weights[:first_H * D_in].T.reshape((-1, first_H, D_in))
        b = weights[first_H * D_in:first_H * D_in + first_H].T.reshape((-1, first_H, 1))
        input = self.h(np.matmul(W, x) + b)
        index = first_H * D_in + first_H

        assert input.shape[1] == first_H

        # additional hidden layers
        # looping through each hidden layer
        # zipped to simultaneously loop through all hidden Hs
        for i in range(0,len(H)-1):
            before = index
            W = weights[index:index + H[i] * H[i+1]].T.reshape((-1, H[i+1], H[i]))
            index += H[i] * H[i+1]
            b = weights[index:index + H[i+1]].T.reshape((-1, H[i+1], 1))
            index += H[i+1]
            output = np.matmul(W, input) + b
            input = self.h(output)

            assert input.shape[1] == H[i+1]

        #output layer
        last_H = H[-1]
        W = weights[index:index + last_H * D_out].T.reshape((-1, D_out, last_H))
        b = weights[index + last_H * D_out:].T.reshape((-1, D_out, 1))
        linear_output = np.matmul(W, input) + b
        output = self.params['output_activation_fn'](linear_output)
        assert output.shape[1] == self.params['D_out']

        return output
    
    
    def make_objective(self, x_train, y_train, reg_param):
        if self.params['task'] == 'regression':
            return self.regression_objective(x_train, y_train, reg_param)
        elif self.params['task'] == 'classification':
            return self.classification_objective(x_train, y_train)


    def fit(self, x_train, y_train, params, reg_param=None):
        ''' Wrapper for MLE through gradient descent '''
        assert x_train.shape[0] == self.params['D_in']
        assert y_train.shape[0] == self.params['D_out']

        ### make objective function for training
        self.objective, self.gradient = self.make_objective(x_train, y_train, reg_param)

        ### set up optimization
        step_size = 0.01
        max_iteration = 5000
        check_point = 100
        weights_init = self.weights.reshape((1, -1))
        mass = None
        optimizer = 'adam'
        random_restarts = 5

        if 'step_size' in params.keys():
            step_size = params['step_size']
        if 'max_iteration' in params.keys():
            max_iteration = params['max_iteration']
        if 'check_point' in params.keys():
            self.check_point = params['check_point']
        if 'init' in params.keys():
            weights_init = params['init']
        if 'call_back' in params.keys():
            call_back = params['call_back']
        if 'mass' in params.keys():
            mass = params['mass']
        if 'optimizer' in params.keys():
            optimizer = params['optimizer']
        if 'random_restarts' in params.keys():
            random_restarts = params['random_restarts']


    
        def call_back(weights, iteration, g):
            ''' Actions per optimization step '''
            objective = self.objective(weights, iteration)
            self.objective_trace = np.vstack((self.objective_trace, objective))
            self.weight_trace = np.vstack((self.weight_trace, weights))
            if iteration % check_point == 0:
                print(f"Iteration {iteration} lower bound {objective:.4f}; gradient mag: {np.linalg.norm(self.gradient(weights, iteration)):.4f}")

        ### train with random restarts
        optimal_obj = 1e16
        optimal_weights = self.weights

        for i in range(random_restarts):
            if optimizer == 'adam':
                opt_init, opt_update, get_params = adam(step_size=step_size)
                opt_state = opt_init(params)
                
                def step(step, opt_state):
                    weights = get_params(opt_state)
                    objective, grads = jax.value_and_grad(self.objective)(weights)
                    self.objective_trace = np.vstack((self.objective_trace, objective))
                    self.weight_trace = np.vstack((self.weight_trace, weights))
                    opt_state = opt_update(step, grads, opt_state)
                    if step % check_point == 0:
                        print(f"Iteration {step} lower bound {objective:.4f}; gradient mag: {np.linalg.norm(grads):.4f}")
                    return value, opt_state

                for step in range(max_iteration):
                    value, opt_state = step(step, opt_state)
                
                #adam(self.gradient, weights_init, step_size=step_size, num_iters=max_iteration, callback=call_back)
            local_opt = np.min(self.objective_trace[-100:])
            
            if local_opt < optimal_obj:
                opt_index = np.argmin(self.objective_trace[-100:])
                self.weights = self.weight_trace[-100:][opt_index].reshape((1, -1))
            weights_init = self.random.normal(0, 1, size=(1, self.D))

        self.objective_trace = self.objective_trace[1:]
        self.weight_trace = self.weight_trace[1:]
        
        
    def regression_objective(self, x_train, y_train, reg_param=None):
        ''' Make objective functions: depending on whether or not you want to apply l2 regularization '''

        if reg_param is None:

            def objective(W):
                squared_error = np.linalg.norm(y_train - self.forward(W, x_train), axis=1)**2
                sum_error = np.sum(squared_error)
                return sum_error

            return objective, grad(objective)

        else:

            def objective(W):
                squared_error = np.linalg.norm(y_train - self.forward(W, x_train), axis=1)**2
                mean_error = np.mean(squared_error) + reg_param * np.linalg.norm(W)
                return mean_error

            return objective, grad(objective)

        
    def classification_objective(self, x_train, y_train):
        def objective(W):
            # From https://github.com/scikit-learn/scikit-learn/blob/0fb307bf39bbdacd6ed713c00724f8f871d60370/sklearn/neural_network/_base.py#L226
            y_prob = self.forward(W, x_train)
            eps = np.finfo(y_prob.dtype).eps
            y_prob = np.clip(y_prob, eps, 1 - eps)
            
            value = -(xlogy(y_train, y_prob) +
                xlogy(1 - y_train, 1 - y_prob)).sum() / y_prob.shape[0]
            return value

        return objective, grad(objective)
    
def xlogy(x, y):
    # Not fast like the scipy version
    return x * np.log(y)