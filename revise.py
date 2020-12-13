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
            calc_loss = _binary_crossentropy
        self._calc_loss = calc_loss
        self._target_class = target_class

        
    def shortest_path_to_target_class(
        self, chosen_point, learning_rate=1e-3, dist_weight=1e-5, 
        max_iter=200, min_prob_of_target=0.5, calc_dist=None):
        
        if calc_dist == None:
            calc_dist = _euclidean
            
        new_z = self.vae.encode(chosen_point)[:self.vae.n_latent_dims]

        i = 0
        reconstructed_prob = 0
 
        pbar = tqdm(total=max_iter, smoothing=0)
        grad_objective_wrt_z = grad(_revise_objective, 1, has_aux=True)
        path = Path(
            x=[chosen_point],
            z=[new_z],
            prob=[self.classifier.predict(chosen_point)],
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
        
        
    def show_path(self, path, dataset=None, zoom=False, landscape='loss', ax=None, fig=None):
        '''Assumes two-dimensional space'''
        landscape_fns = {
            'loss': self._loss_landscape,
            'likelihood': self._likelihood_landscape,
            'prob_target': self._prob_target_landscape
        }
        grid, positions = self._grid_positions(dataset)
        x, y = grid
        loss = landscape_fns[landscape](positions).reshape(grid.shape[1:3])
        titles = {
            'loss': 'REVISE path against REVISE objective',
            'likelihood': 'REVISE path against data log-likelihood under VAE',
            'prob_target': 'REVISE path against classifier probability of target class'
        }
        title = titles[landscape]
        chosen_point = path.x[0]
        if ax is None:
            fig, ax = plt.subplots(1,1)
        contour = ax.contourf(x, y, loss, levels=100 if zoom else 50, cmap='viridis')
        fig.colorbar(contour, ax=ax)
        if dataset is not None:
            ax.scatter(dataset.T[0], dataset.T[1], color='white', alpha=0.6)
        ax.scatter(*chosen_point, color='red')
        x_path = np.array(path.x)
        if zoom:
            ax.set_xlim(x_path[:,0].min() - .1, x_path[:,0].max() + .1)
            ax.set_ylim(x_path[:,1].min() - .1, x_path[:,1].max() + .1)
            title += ' (zoomed)'
        #iter_by = (x_path.shape[0] // 10) + 1
        #plt.plot(x_path[::iter_by,0], x_path[::iter_by,1], c='red', marker='o', alpha=0.7)
        ax.set_title(title)
        ax.plot(x_path[:,0], x_path[:,1], c='red', alpha=0.5, marker='.')
        return ax

        
    def _loss_landscape(self, positions):
        '''Assumes two-dimensional space'''
        prob_target = self.classifier.predict(positions)
        target = numpy.ones_like(prob_target) * self._target_class
        binary_crossentropy = -(xlogy(target, prob_target) + xlogy(1 - target, 1 - prob_target))
        return binary_crossentropy
    
    
    def _likelihood_landscape(self, positions):
        z = self.vae.encode(positions)
        new_x = self.vae.decode(z[:,:2])
        log_prob = numpy.sum(norm.logpdf(positions, new_x, self.vae.x_var ** 0.5), axis=1)
        return log_prob
    
    
    def _prob_target_landscape(self, positions):
        prob_target = self.classifier.predict(positions)
        return prob_target
    
    
    def _grid_positions(self, dataset=None, grid=None):
        if grid is None and dataset is None:
            grid = numpy.mgrid[-5:5:0.2,-5:5:0.2]
        if dataset is not None:
            x_lim = (dataset[:,0].min(), dataset[:,0].max())
            y_lim = (dataset[:,1].min(), dataset[:,1].max())
            x_margin = (x_lim[1]-x_lim[0]) * 0.2
            y_margin = (y_lim[1]-y_lim[0]) * 0.2
            grid = numpy.mgrid[
                x_lim[0]-x_margin:x_lim[1]+x_margin:(x_lim[1]-x_lim[0])/100,
                y_lim[0]-y_margin:y_lim[1]+y_margin:(y_lim[1]-y_lim[0])/100]
        x, y = grid
        return grid, numpy.vstack((x.flatten(), y.flatten())).T
    
    
    def _calc_loss_at_point(self, point, target):
        prob_target = self.classifier.predict(numpy.array(point))
        return self._calc_loss(target, prob_target)

        
def _revise_objective(chosen_point, chosen_z, vae, classifier, calc_loss, dist_weight, calc_dist, target_class):
    reconstructed_x = vae.decode(chosen_z)
    distance = calc_dist(chosen_point, reconstructed_x)
    distance_term = dist_weight * distance
    reconstructed_prob = classifier.predict(reconstructed_x)
    loss = calc_loss(np.array([1]), reconstructed_prob)
    objective = loss + distance_term
    return objective, (reconstructed_x, reconstructed_prob)


def _euclidean(x1, x2):
    return np.linalg.norm(x1 - x2, ord=2)


def _binary_crossentropy(true, prob):
    return -(xlogy(true, prob) + xlogy(1 - true, 1 - prob)).sum() / prob.shape[0]