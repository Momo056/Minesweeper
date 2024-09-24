from itertools import product
import json
from random import sample


class Hyperparameter_Journal:
    def __init__(self, save_file: str) -> None:
        self.save_file = save_file
        self.init_record()
        self.records: dict

    def init_record(self):
        try:
            with open(self.save_file, 'r') as file:
                self.records = json.load(file)
            self.compute_remainings()
        except FileNotFoundError:
            self.records = {}

    def first_load(self, hparams_domain: dict[str, list]):
        self.records['domain'] = hparams_domain
        self.records['keys'] = sorted(list(hparams_domain.keys()))
        self.records['x'] = []
        self.records['y'] = []
        self.records['transform_domain'] = [len(hparams_domain[k]) for k in self.records['keys']]
        self.compute_remainings()
        self.save()

    def save(self):
        with open(self.save_file, 'w') as file:
            json.dump(self.records, file, indent=4)
    
    def compute_remainings(self):
        all_combinations = set(product(*[range(l) for l in self.records['transform_domain']]))
        for x in self.records['x']:
            all_combinations.remove(x)
        self.remainings = list(all_combinations)

    def get_xy(self):
        return self.records['x'], self.records['y']
    
    def to_hparam(self, x_transformed):
        return {k:self.records['domain'][k][i] for k, i in zip(self.records['keys'], x_transformed)}
    
    def record_xy(self, x, y):
        self.remainings.remove(x)
        self.records['x'].append(x)
        self.records['y'].append(y)
        self.save()

    def sample_x(self, n_sample: int):
        return sample(self.remainings, n_sample)
    
    def __len__(self):
        return len(self.records['x'])

