#!/usr/bin/env python

"""
Auxiliary functions to write ferroX input files
"""

def generate_input_from_default(**kwargs):
    simulation_parameters_dictionary = { 
        "domain.prob_lo" :  [-16.e-9, -16.e-9, 0.e-9],
        "domain.prob_hi" :   [16.e-9,  16.e-9, 16.e-9], # dx = dy != dz (more resolution in the z direction)
        "domain.n_cell" : [64, 64, 32],
        "domain.max_grid_size" :  [64, 64, 32],
        "domain.blocking_factor" :  [32, 32, 32],
        "domain.coord_sys" :  'cartesian',
        "prob_type" : 2,
        "TimeIntegratorOrder" :  1,
        "amrex.the_arena_is_managed" : 1,
        "nsteps" : 150000,
        "plot_int" : 150001,
        "dt" : 1.0e-13,
        "P_BC_flag_lo" :  [3, 3, 0],
        "P_BC_flag_hi" :  [3, 3, 1],
        "lambda" :  3.0e-9,
        "domain.is_periodic" : [1, 1, 0],
        "boundary.hi" : 'per per dir(0.0)',
        "boundary.lo" : 'per per dir(0.0)',
        "voltage_sweep" : 1,
        "Phi_Bc_lo" : 0.0,
        "Phi_Bc_hi" : 0.0,
        "Phi_Bc_inc" : 0.05,
        "Phi_Bc_hi_max" :  1.0,
        "num_Vapp_max": 21, 
        "phi_tolerance" :  1.e-5,
        "FE_lo" :  [-16.e-9, -16.e-9, 11.0e-9],
        "FE_hi" :   [16.e-9,  16.e-9, 16.0e-9],
        "DE_lo" :  [-16.e-9, -16.e-9, 10.0e-9],
        "DE_hi" : [16.e-9 , 16.e-9, 11.0e-9],
        "SC_lo" :  [-16.e-9, -16.e-9 , 0.0e-9],
        "SC_hi" :  [16.e-9, 16.e-9 ,10.0e-9],
        "epsilon_0" :  8.85e-12,
        "epsilonX_fe" :  24.0,
        "epsilonZ_fe" :  24.0,
        "epsilon_de" :  3.9,
        "epsilon_si" :  11.7,
        "alpha" : -2.5e9,
        "beta" :  6.0e10,
        "gamma" :  1.5e11,
        "BigGamma" : 100,
        "g11" : 1.0e-9,
        "g44" :  1.0e-9,
        "g44_p" :  0.0,
        "g12" :  0.0,
        "alpha_12" :  0.0,
        "alpha_112" :  0.0,
        "alpha_123" :  0.0,
        "acceptor_doping" : 1e20}
    for key, value in kwargs.items():
        simulation_parameters_dictionary[key] = value 
        
    return simulation_parameters_dictionary

def generate_input_lines(param_dict):
    lines = []
    for key in param_dict.keys():
        first_part = str(key) + " = "
        value = param_dict[key]
        if isinstance(value, str):
            second_part = str(value)
        if isinstance(value, float):
            second_part = "{:.2e}".format(value)
        if isinstance(value, list):
            if isinstance(value[0], int):
                second_part = "{} {} {}".format(value[0], value[1], value[2])
            if isinstance(value[0], float):
                second_part = "{:.2e} {:.2e} {:.2e}".format(value[0], value[1], value[2])
        if isinstance(value, int):
            second_part = str(value)
        lines.append(first_part + second_part + str('\n'))
    return lines

def write_input_from_default(simulation_path, **kwargs):
    simulation_parameters_dictionary = generate_input_from_default(**kwargs)
    input_lines = generate_input_lines(param_dict=simulation_parameters_dictionary)
    f = open(simulation_path+"inputc", "w")
    f.writelines(input_lines)
    f.close()