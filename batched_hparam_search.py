
from math import prod
from os import makedirs, path
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



def get_custom_cfg(x: list, journal: Hyperparameter_Journal, batch_idx: int, batch_container: str):
    if batch_idx == 0:
        cfg = YAML_Configuration()
    else:
        cfg = Configuration(path.join(batch_container, 'best_configs', f'batch-{batch_idx-1}.cfg'))
    hparams = journal.to_hparam(x)
    cfg.data.update(hparams)
    return cfg

def get_sampler(journal: Hyperparameter_Journal):
    # Define the sampler
    all_x, all_y = journal.get_xy()
    sampler = Optimized_Sampler(journal, 100)
    sampler.set_gaussian_process(GaussianProcessRegressor(
        kernel=ConstantKernel()*Matern(length_scale=np.ones((len(all_x[0]),)), nu=2.5),
        normalize_y=True,
        n_restarts_optimizer=10,
    ))
    return sampler

if __name__ == '__main__':
    max_error = 10
    nb_error = 0

    yml_cfg = YAML_Configuration()

    batch_param_container = 'batch_hparams_runs'

    for batch_idx, batch_param in enumerate(yml_cfg.INDEPENDANT_BATCHS):
        print(f'Start batch {batch_idx}')
        print(batch_param)
        # Build if necessary journal
        journal = Hyperparameter_Journal(path.join(batch_param_container, 'journals', f'{batch_idx}.yml'))

        try:
            len(journal)
        except: # Journal is not loaded
            # Construct hparam dict
            hparam_domain = {
                key: yml_cfg.get_ordered(key)
                for key in batch_param
            }
            print(hparam_domain)
            journal.first_load(hparam_domain)

        # Initialize the sampler variable
        sampler = None

        # Create if necessary paths for run container
        run_container = path.join(batch_param_container, 'runs', f'batch-{batch_idx}')
        makedirs(run_container, exist_ok=True)

        # Build runner
        runner = Hyperparameter_Runner(run_container)

        # Optimize
        while len(journal) < 32 and len(journal.remainings) > 0:
            # Select hyper-parameter
            if len(journal) > 0:
                if sampler is None:
                    sampler = get_sampler(journal)

                x = sampler.next_x()
            else:
                x, = journal.sample_x(1)

            # Preapre the run
            run_cfg = get_custom_cfg(x, journal, batch_idx, batch_param_container)

            print(f'Training {len(journal)+1}')
            print(f'Index form : {x}')

            print(run_cfg.get_as_string())

            # Run
            try:
                print(f'Start training {len(journal)+1}')
                result = runner.run(run_cfg)
                y = result['test/accuracy']
                print(f'{y=}')
                journal.record_xy(x, y)
                nb_error += max_error
            except RuntimeError:
                print(f'{nb_error=}')
                nb_error += 1

        # Log
        x_list, y_list = journal.get_xy()
        best_i = np.argmax(y_list)
        best_cfg = get_custom_cfg(x_list[best_i], journal, batch_idx, batch_param_container)
        best_cfg.save_to(path.join(batch_param_container, 'best_configs', f'batch-{batch_idx}.cfg'))

print(f'End of script : {nb_error=}')





