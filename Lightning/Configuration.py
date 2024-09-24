

import ast
from typing import Any


CFG_TEMPLATE_PATH = 'Lightning/config.cfg'

class Configuration:
    def __init__(self, cfg_path: str = CFG_TEMPLATE_PATH) -> None:
        self.data = {}
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

    def __getattribute__(self, name: str) -> Any:
        if name == 'data' or name not in self.data.keys():
            return super().__getattribute__(name)
        return self.data[name]
        
    def __setattr__(self, name: str, value: Any) -> None:
        if name == 'data' or name not in self.data.keys():
            return super().__setattr__(name, value)
        self.data[name] = value
    
    def get_as_string(self):
        with open(self.cfg_path, 'r') as file:
            lines = [l.strip() for l in file.readlines()]

        new_lines = []
        to_do_keys = {*self.data.keys()}
        for line in lines:
            to_add, changed_key = self._str_line(line)
            if changed_key is not None:
                to_do_keys.remove(changed_key)
            new_lines.append(to_add)

        new_lines.append('')
        for k in to_do_keys:
            new_lines.append(self._str_key_value(k, self.data[k]))

        return '\n'.join(new_lines)

    def _str_line(self, line: str):
        split_equal = line.find('=')
        if split_equal == -1 or line.startswith('#'):
            return line, None
        key = line[:split_equal].strip(' ')

        if key in self.data.keys():
            return self._str_key_value(key, self.data[key]), key
        return line, None
    
    def _str_key_value(self, key, value):
        if isinstance(value, str):
            value = f'"{value}"'
        return f'{key} = {value}'
    
    def save_to(self, save_path: str):
        with open(save_path, 'w') as file:
            file.write(self.get_as_string())


