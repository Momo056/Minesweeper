import json
from os import listdir, makedirs, path
import subprocess
import sys
from typing import Any
from Lightning.Configuration import Configuration


class Hyperparameter_Runner:
    def __init__(self, container_dir: str = 'hyperparameters_runs', redirect_command:bool = True) -> None:
        self.container = container_dir
        self.executable = sys.executable
        self.redirect_command = redirect_command

        self.load_current_id()
        self.current_id: int

    def load_current_id(self):
        dir_ids = []
        for d in listdir(self.container):
            try:
                dir_ids.append(int(d))
            except ValueError:
                pass
        
        if len(dir_ids) == 0:
            self.current_id = 0
            return

        self.current_id = max(dir_ids)

    def pop_run_directory(self):
        self.current_id += 1
        run_dir = path.join(self.container, f'{self.current_id}') 
        makedirs(run_dir)
        return run_dir

    
    def get_command_arguments(self, run_directory: str):
        return path.join(run_directory, 'config.cfg'), run_directory, path.join(run_directory, 'result.json')
    
    def get_redirect_file(self, run_directory: str):
        return path.join(run_directory, 'out'), path.join(run_directory, 'err')
    
    def get_command(self, config_path, run_container, run_output_file):
        return f'{self.executable} ./training_script.py --config_file {config_path} --run_dir_container {run_container} --output_file {run_output_file}'
        
    def run(self, config: Configuration) -> dict[str, Any]:
        # Path and directory preparations
        run_dir = self.pop_run_directory()

        config_path, run_container, run_output_file = self.get_command_arguments(run_dir)

        # Prapare config file passed to the training script
        config.save_to(config_path)

        # Command
        command = self.get_command(config_path, run_container, run_output_file)
        if self.redirect_command:
            out_path, err_path = self.get_redirect_file(run_dir)
            with open(out_path, 'w') as file_out:
                with open(err_path, 'w') as file_err:
                    subprocess.run(command, stdout=file_out, stderr=file_err)
        else:
            subprocess.run(command)
        
        with open(run_output_file, 'r') as file:
            result = json.load(file)

        return result




