#!/usr/bin/env python

"""
This script is responsible for iterating over the variables that will be modified in an experiment,
creating a directory for each combination of interest with all the files necessary to run an individual
simulation in that directory, and submitting the job to the Slurm worload manager.

The user must provide code to modify the variables of interest in sim_params_dict and sbatch_options_dict,
as well as a description of the experiment. The function new_simulation_from_default should be called
once for each simulation. The script should be executed from the command line. 
"""

import numpy as np

from utils.new_simulation_from_default import new_simulation_from_default
from utils.dir_structure_utils import find_next_available_file
from utils.config import simulations_path, executables_path, executable, experiments_path

if __name__ == "__main__":

    simulation_ids = []

#### SCRIPT SHOULD BE BELOW ####      
    
    experiment_description = """This experiment assigns the values [1.0nm, 1.5nm, 2.0nm] to the
        thicknesses to the dielectric and ferroelectro layers of a FerroX stack and creates a
        directory for each combination (total of 9) with all the files necessary to run each simulation.
        It keeps the domain cell size constant."""
    
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

            simulation_id = new_simulation_from_default(simulations_path, executables_path, executable, 
                                                        sim_params_dict, sbatch_options_dict, submit_job=False)
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