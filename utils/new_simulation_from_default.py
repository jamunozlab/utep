#!/usr/bin/env python

"""
Sets up a new simulation given a simulations path, executable path and filename,
and input and sbatch parameters to change from a default.
"""

import os
from .dir_structure_utils import find_next_available_dir
from .write_input_from_default import write_input_from_default
from .write_run_sbatch_from_default import write_run_sbatch_from_default

def new_simulation_from_default(simulations_path, executables_path, executable, sim_params_dict, sbatch_options_dict, submit_job=True):
    os.chdir(simulations_path)
    simulation_id = find_next_available_dir('simulation', simulations_path)
    simulation_path = simulations_path + simulation_id + '/'
    os.mkdir(simulation_path)
    executable_path = executables_path + executable
    new_executable_path = simulation_path + executable
    command = 'ln -s ' + executable_path + ' ' + new_executable_path
    os.system(command)
    write_input_from_default(simulation_path, **sim_params_dict)
    write_run_sbatch_from_default(simulation_path, executable, **sbatch_options_dict)
    print('New simulation is in ' + simulation_path)
    
    if submit_job:
        os.chdir(simulation_path)
        os.system('sbatch ./runc.sbatch')
        
    return simulation_id