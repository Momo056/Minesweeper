
from math import prod
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from tqdm import tqdm
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern
import yaml

from Lightning.Configuration import Configuration
from Lightning.YAML_Configuration import YAML_Configuration
from hyperparameter_tunning.Hyperparameter_Journal import Hyperparameter_Journal
from hyperparameter_tunning.Hyperparameter_Runner import Hyperparameter_Runner
from hyperparameter_tunning.Optimized_Sampler import Optimized_Sampler


# Consider that the procimity of args is the distance of their indexs
# ex : For LATENT_DIM, distance(16, 128) -> 3  ((index 4 - index 1))
hparams_domain = {
    'ALPHA' : [0.5, 0.7, 0.9, 1.0],
    'N_SYM_BLOCK' : [1, 2, 3, 4],
    'LAYER_PER_BLOCK' : [1, 2, 3, 4],
    'LATENT_DIM' : [16, 32, 64, 128],
    'KERNEL_SIZE' : [3, 5, 7, 9],
    'BATCH_NORM_PERIOD' : [1, 2, 4, 0],
    'LR': [1e-2, 1e-3, 1e-4], # , 1e-5, 1e-6],
}

journal = Hyperparameter_Journal('hyperparameter_tunning/journal.yml')
try:
    len(journal)
except:
    journal.first_load(hparams_domain)

sampler = None
def get_sampler(journal):
    # Define the sampler
    all_x, all_y = journal.get_xy()
    sampler = Optimized_Sampler(journal, 100)
    sampler.set_gaussian_process(GaussianProcessRegressor(
        kernel=ConstantKernel()*Matern(length_scale=np.ones((len(all_x[0]),)), nu=2.5),
        normalize_y=True,
        n_restarts_optimizer=10,
    ))
    return sampler

runner = Hyperparameter_Runner()

random_guesses = 10

# if len(journal) < random_guesses:
#     next_guesses = journal.sample_x(random_guesses - len(journal))
# else:
#     raise NotImplementedError()

print('Start Hyperparameter exploration')
nb_error = 0
max_error = 20
while nb_error < max_error:
    if len(journal) > 0:
        if sampler is None:
            sampler = get_sampler(journal)

        x = sampler.next_x()
    else:
        x, = journal.sample_x(1)


    cfg = YAML_Configuration()

    hparams = journal.to_hparam(x)
    cfg.data.update(hparams)

    print(f'Training {len(journal)+1}')
    print(hparams)
    print(f'Index form : {x}')

    print()
    print(cfg.get_as_string())

    try:
        # result = runner.run(cfg)
        y = result['test/accuracy']
        print(f'{y=}')
        journal.record_xy(x, y)
        nb_error += max_error
    except RuntimeError:
        print(f'{nb_error=}')
        nb_error += 1

print(f'End of script : {nb_error=}')

