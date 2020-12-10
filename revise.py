from collections import namedtuple
from functools import partial, cached_property

from jax import grad
import jax.numpy as np
from jax.scipy.special import xlogy
import matplotlib.pyplot as plt
import numpy
from scipy.stats import norm
from tqdm.auto import tqdm

Path = namedtuple('Path', ['x', 'z', 'prob', 'grad'], defaults=[[],[],[],[]])

class Revise:
    def __init__(self, classifier, vae, calc_loss=None, target_class=1):
        self.classifier = classifier
        self.vae = vae
        if calc_loss is None:
            calc_loss = self._binary_crossentropy
        self._calc_loss = calc_loss
        self._target_class = target_class

        
    def shortest_path_to_target_class(
        self, chosen_point, learning_rate=1e-3, dist_weight=1e-5, 
        max_iter=200, min_prob_of_target=0.5, calc_dist=None):
        
        if calc_dist == None:
            calc_dist = self._euclidean
            
        chosen_point = chosen_point.T
        new_z = self.vae.infer(chosen_point.reshape(1,-1).T)[:self.vae.z_dim,0]

        i = 0
        reconstructed_prob = 0
 
        pbar = tqdm(total=max_iter, smoothing=0)
        grad_objective_wrt_z = grad(self._revise_objective, 1, has_aux=True)
        path = Path(
            x=[chosen_point],
            z=[new_z],
            prob=[self.classifier.predict_proba(chosen_point.reshape(1, -1))],
            grad=[])
        
        while i < max_iter and reconstructed_prob < min_prob_of_target:
            grad_z, (reconstructed_x, reconstructed_prob) = grad_objective_wrt_z(
                chosen_point, new_z, self.vae, self.classifier, self._calc_loss,
                dist_weight, calc_dist, self._target_class)
            new_z -= learning_rate * grad_z
            path.x.append(reconstructed_x)
            path.z.append(new_z)
            path.prob.append(reconstructed_prob)
            path.grad.append(grad_z)
            pbar.update()
            i += 1
        pbar.close()
        return path
        
        
    def show_path(self, path, dataset=None, zoom=False, landscape='loss'):
        '''Assumes two-dimensional space'''
        landscape_fns = {
            'loss': self._loss_landscape,
            'likelihood': self._likelihood_landscape,
            'prob_target': self._prob_target_landscape
        }
        (x, y), loss = landscape_fns[landscape]
        titles = {
            'loss': 'REVISE path against REVISE objective',
            'likelihood': 'REVISE path against data log-likelihood under VAE',
            'prob_target': 'REVISE path against classifier probability of target class'
        }
        title = titles[landscape]
        chosen_point = path.x[0]
        plt.contourf(x, y, loss, levels=100 if zoom else 30, cmap='viridis')
        plt.colorbar()
        if dataset is not None:
            plt.scatter(dataset.T[0], dataset.T[1], color='white', alpha=0.6)
        plt.scatter(*chosen_point, color='red')
        x_path = np.array(path.x)
        if zoom:
            plt.xlim(x_path[:,0].min() - .1, x_path[:,0].max() + .1)
            plt.ylim(x_path[:,1].min() - .1, x_path[:,1].max() + .1)
            title += ' (zoomed)'
        #iter_by = (x_path.shape[0] // 10) + 1
        #plt.plot(x_path[::iter_by,0], x_path[::iter_by,1], c='red', marker='o', alpha=0.7)
        plt.title(title)
        plt.plot(x_path[:,0], x_path[:,1], c='red', alpha=0.5, marker='.')
        plt.show()

        
    @cached_property
    def _loss_landscape(self, grid=None):
        '''Assumes two-dimensional space'''
        grid, positions = self._grid_positions(grid)
        prob_target = self.classifier.predict_proba(positions)
        target = numpy.ones_like(prob_target) * self._target_class
        binary_crossentropy = -(xlogy(target, prob_target) + xlogy(1 - target, 1 - prob_target)).reshape(grid.shape[1:3])
        return grid, binary_crossentropy
    
    
    @cached_property
    def _likelihood_landscape(self, grid=None):
        grid, positions = self._grid_positions(grid)
        z = self.vae.encoder.forward(self.vae.encoder.weights, positions.T).squeeze()
        new_x = self.vae.decoder.forward(self.vae.decoder.weights, z[:2,:]).squeeze().T
        log_prob = numpy.sum(norm.logpdf(new_x, positions, self.vae.x_var ** 0.5), axis=1).reshape(grid.shape[1:3])
        return grid, log_prob
    
    
    @cached_property
    def _prob_target_landscape(self, grid=None):
        grid, positions = self._grid_positions(grid)
        prob_target = (self.classifier.predict_proba(positions)).reshape(grid.shape[1:3])
        return grid, prob_target
    
    
    def _grid_positions(self, grid=None):
        if grid is None:
            grid = numpy.mgrid[-5:5:0.2,-5:5:0.2]
        x, y = grid
        return grid, numpy.vstack((x.flatten(), y.flatten())).T
    
    
    def _calc_loss_at_point(self, point, target):
        prob_target = self.classifier.predict_proba(numpy.array(point).reshape(1, -1)).reshape(1,-1)
        return self._calc_loss(target, prob_target)
        
        
    @staticmethod
    def _revise_objective(chosen_point, chosen_z, vae, classifier, calc_loss, dist_weight, calc_dist, target_class):
        reconstructed_x = vae.decoder.forward(vae.decoder.weights.reshape(1,-1), chosen_z.reshape(1,-1).T).reshape(1,-1).squeeze()
        distance = calc_dist(chosen_point, reconstructed_x)
        distance_term = dist_weight * distance
        reconstructed_prob = classifier.predict_proba(reconstructed_x.reshape(1, -1))
        loss = calc_loss(np.array([1]), reconstructed_prob.reshape(1,-1))
        objective = loss + distance_term
        return objective, (reconstructed_x, reconstructed_prob)
    
    
    @staticmethod
    def _euclidean(x1, x2):
        return np.linalg.norm(x1 - x2, ord=2)
    
    
    @staticmethod
    def _binary_crossentropy(true, prob):
        return -(xlogy(true, prob) + xlogy(1 - true, 1 - prob)).sum() / prob.shape[0]