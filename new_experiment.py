#!/usr/bin/env python

"""
This script is responsible for iterating over the variables that will be modified in an experiment,
creating a directory for each combination of interest with all the files necessary to run an individual
simulation in that directory, and submitting the job to the Slurm worload manager.

The user must provide code to modify the variables of interest in sim_params_dict and sbatch_options_dict,
as well as a description of the experiment. The function new_simulation_from_default should be called
once for each simulation. The script should be executed from the command line. 
"""

import argparse
import numpy as np

from utils.new_simulation_from_default import new_simulation_from_default
from utils.dir_structure_utils import find_next_available_file
from utils.config import simulations_path, executables_path, executable, experiments_path, other_files
import utils.default_dictionaries as default_dictionaries

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='new_experiment',
                    description='Create new experiment. User must provide the name of a dictionary' +
                                    'in default_dictionaries.py with input parameters and default values.',
                    epilog='')
    
    parser.add_argument('default_input_dictionary_name') # positional argument
    parser.add_argument('-t', '--test', action='store_false')  # on/off flag
    
    submit_job=True

    args = parser.parse_args()
    submit_job = args.test
    exec('default_input_dictionary = default_dictionaries.' + args.default_input_dictionary_name)
    
    simulation_ids = []


#### SCRIPT SHOULD BE BELOW ####      
    
    experiment_description = """Test of MagneX"""
    
    nsteps = [40000, 4000, 400]
    
    count = 0
    for nstep in nsteps:
        sim_params_dict = {  
               "nsteps"   :   nstep,
             }
    
        sbatch_options_dict = { }
        count = count + 1

        simulation_id = new_simulation_from_default(default_input_dictionary, simulations_path, executables_path, executable,
                                                        sim_params_dict, sbatch_options_dict, other_files=other_files, submit_job=submit_job)
        simulation_ids.append(simulation_id)
        
    experiment_id = find_next_available_file('experiment', experiments_path)

#### SCRIPT SHOULD BE ABOVE ####  

    lines = experiment_description + str('\n')
    for simulation_id in simulation_ids:
        lines = lines + simulation_id + str('\n')

    f = open(experiments_path+experiment_id, "w")
    f.writelines(lines)
    f.close()
    
    print(experiment_id)
    print(simulation_ids)
    
    
""" 
    DE_loz = 10.0e-9
    incrs = [1.0, 1.5, 2.0]
    
    count = 0
    for inc in incrs:
        base = 10 + inc
        DE_hiz = DE_loz + inc * 1.0e-9
        FE_loz = DE_hiz
        for inc in incrs:
            base = base + inc 
            FE_hiz = FE_loz #+ inc * 1.0e-9 
               
        
            sim_params_dict = {  
               "domain.prob_hi"   :   [16.e-9,  16.e-9, FE_hiz],
               "DE_lo"            : [-16e-9,-16e-9, DE_loz],
               "DE_hi"            : [16e-9, 16e-9, DE_hiz],
               "FE_lo"            : [-16e-9, -16e-9, FE_loz],
               "FE_hi"            : [16e-9, 16e-9, FE_hiz],  
               "domain.n_cell"           : [64, 64, int(np.round(FE_hiz/0.25e-9,0))],
               "domain.max_grid_size"    : [64, 64, int(np.round(FE_hiz/0.25e-9,0))],
               "domain.blocking_factor"  : [64, 64, int(np.round(FE_hiz/0.25e-9,0))],
                
             }
        
            sbatch_options_dict = { }
            count = count + 1

            simulation_id = new_simulation_from_default(default_input_dictionary, simulations_path, executables_path, executable,
                                                        sim_params_dict, sbatch_options_dict, submit_job=submit_job)
            simulation_ids.append(simulation_id)
        
    experiment_id = find_next_available_file('experiment', experiments_path)
"""