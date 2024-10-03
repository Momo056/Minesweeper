import ast
from typing import Any

class Abstract_Configuration:
    def __init__(self) -> None:
        self.data = {}

    def __getattribute__(self, name: str) -> Any:
        if name == 'data' or name not in self.data.keys():
            return super().__getattribute__(name)
        return self.data[name]
    
    def get_template(self):
        return ''
    
    def get_as_string(self):
        lines = [l.strip() for l in self.get_template().split('\n')]

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

        return '\n'.join(new_lines).strip()

    def _str_line(self, line: str):
        split_equal = line.find('=')
        if split_equal == -1 or line.startswith('#'): # Not a key-value entry
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
 