from typing import Callable
from numpy import ndarray
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern
from hyperparameter_tunning.Hyperparameter_Journal import Hyperparameter_Journal
import numpy as np


class Optimized_Sampler:
    def __init__(self, journal: Hyperparameter_Journal, n_sample: int) -> None:
        self.journal = journal
        self.n_sample = n_sample
        self.set_gaussian_process(GaussianProcessRegressor(
            kernel=ConstantKernel()*RBF(),
            normalize_y=True,
        ))
        self.set_acquisition_function(lambda mu, sigma: mu + sigma)

    def set_gaussian_process(self, gaussian_process: GaussianProcessRegressor):
        self.gaussian_process = gaussian_process
        return self
    
    def set_acquisition_function(self, acquisition_function: Callable[[ndarray, ndarray], ndarray]):
        # acquisition_function : (mu, sigma) -> score
        # The should is maximized
        self.acquisition_function = acquisition_function
        return self
    
    def sample_sorted(self):
        # Sample input space
        x_sample = self.journal.sample_x(self.n_sample)

        # Predict their output
        x, y = self.journal.get_xy()
        self.gaussian_process.fit(x, y)
        y_predict, std_predict = self.gaussian_process.predict(x_sample, return_std=True)
        scores = self.acquisition_function(y_predict, std_predict)

        # Sort from best to worst
        i_sorted = np.argsort(scores)[::-1]

        # Return sorted
        return np.array(x_sample)[i_sorted], np.array(scores)[i_sorted], y_predict[i_sorted], std_predict[i_sorted]

    def next_x(self):
        x, score, y, std = self.sample_sorted()
        return self.format_x(x[0])
    
    def format_x(self, x: ndarray):
        return tuple(x.tolist())
        
        
