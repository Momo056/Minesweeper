import ast
from typing import Any

from Lightning.Abstract_Configuration import Abstract_Configuration


CFG_TEMPLATE_PATH = 'Lightning/config.cfg'

class Configuration(Abstract_Configuration):
    def __init__(self, cfg_path: str = CFG_TEMPLATE_PATH) -> None:
        super().__init__()
        self.cfg_path = cfg_path
        self.load_cfg(cfg_path)

    def load_cfg(self, cfg_path: str):
        with open(cfg_path, 'r') as file:
            lines = [l.strip(' \n') for l in file.readlines()]

        for line in lines:
            if line.startswith('#'):
                pass
            else:
                self._load_line(line)

    def _load_line(self, line: str):
        split_equal = line.find('=')
        if split_equal == -1:
            return

        key = line[:split_equal].strip(' \n')
        value_str = line[split_equal+1:].strip(' \n')
        value = ast.literal_eval(value_str) # Read it as a Python value
        self.data[key] = value

    def get_template(self):
        with open(self.cfg_path, 'r') as file:
            template = file.read()
        return template


