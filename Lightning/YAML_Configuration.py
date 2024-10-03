

import ast
from typing import Any
import numpy as np
import yaml
from Lightning.Abstract_Configuration import Abstract_Configuration
from Lightning.Configuration import CFG_TEMPLATE_PATH as STANDARD_TEMPLATE_PATH


CFG_TEMPLATE_PATH = 'Lightning/config.yml'

class YAML_Configuration(Abstract_Configuration):
    def __init__(self, cfg_path: str = CFG_TEMPLATE_PATH, standard_template_path: str = STANDARD_TEMPLATE_PATH) -> None:
        super().__init__()
        self.cfg_path = cfg_path
        self.standard_template_path = standard_template_path
        self.load_cfg(cfg_path)

    def load_cfg(self, cfg_path: str):
        with open(cfg_path, 'r') as file:
            self.data = yaml.safe_load(file)

    def get_template(self):
        with open(self.standard_template_path, 'r') as file:
            template = file.read()
        return template
    
    def _str_key_value(self, key, value):
        if key == 'INDEPENDANT_BATCHS':
            return ''

        try: # Get the base value if it is a dict
            base_value = value['base']
            value = base_value
        except TypeError:
            pass

        return super()._str_key_value(key, value)
    
    def get_ordered(self, key: str):
        data: dict | Any = self.data[key]

        if 'range' in data.keys():
            start, stop = data['range']

            if 'step' in data.keys():
                step = data['step']
            else:
                step = 1
            
            return [*np.arange(start, stop, step).tolist(), stop]
        
        if 'set' in data.keys():
            return [ast.literal_eval(val) for val in data['set']]
