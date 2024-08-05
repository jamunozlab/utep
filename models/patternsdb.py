# This organizes simulation data in a way that can be used easily during analysis and ml development
# This class is also in /ferrox/ml/models/patternsdb.py
# Once active development ends, it will not be in the ipynb anymore

import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch

from torch.masked import masked_tensor, as_masked_tensor

# Disable prototype warnings and such
#warnings.filterwarnings(action='ignore', category=UserWarning)

class patternsDB_orig(Dataset):
    def __init__(self, simulations_path, *args, train=True): # should be 1 or many simulation directories
        
        self.simulations_path = simulations_path
        self.simulation_paths = []
        for simulation in args:
            simulation_path = simulations_path + simulation
            self.simulation_paths.append(simulation_path)
        
        def get_npz_filenames(simulation_path):
            names = []
            for entry in os.scandir(simulation_path):
                if entry.is_file() and 'npz' in entry.name:
                    names.append(entry.name)
            return names, len(names)
        
        self.list_of_names = []
        self.size = 0
        for simulation_path in self.simulation_paths:
            names, size = get_npz_filenames(simulation_path)
            self.list_of_names.append(names)
            self.size = self.size + len(names)
        
        #self.size = len(self.list_of_names)
        self.idx = np.zeros(self.size)
        self.applied_voltage = np.zeros(self.size)
        self.integrated_charge = np.zeros(self.size)
        self.polarization = np.zeros((64,64,self.size))
        self.efield = np.zeros((64,64,self.size))
    
        self.counter = 0
        for ix in range(len(self.simulation_paths)):
            simulation_path = self.simulation_paths[ix]
            names = self.list_of_names[ix]
            for filename in names:
                npfile = np.load(simulation_path+filename)
                self.idx[self.counter] = int(npfile['ts'])
                self.applied_voltage[self.counter] = float(npfile['applied_voltage'])
                self.integrated_charge[self.counter] = float(npfile['integrated_charge'])
                self.polarization[:, :, self.counter] = npfile['polarization']
                self.efield[:, :, self.counter] = npfile['efield']
                self.counter = self.counter + 1
        
    def __len__(self):
        return self.size

    def __getitem__(self, element):
        #if train and self.it in self.train_ids:
        X = self.efield[:,:,element]
        y = self.integrated_charge[element]
            
        return X, y


class patternsDB(Dataset):
    def __init__(self, simulations_path, *args, **kwargs): # should be 1 or many simulation directories
        
        self.simulations_path = simulations_path
        self.simulation_paths = []
        for simulation in args:
            simulation_path = simulations_path + simulation
            self.simulation_paths.append(simulation_path)
        
        def get_npz_filenames(simulation_path):
            names = []
            for entry in os.scandir(simulation_path):
                if entry.is_file() and 'npz00' in entry.name:
                    names.append(entry.name)
            return names, len(names)
        
        self.list_of_names = []
        self.size = 0
        for simulation_path in self.simulation_paths:
            names, size = get_npz_filenames(simulation_path)
            self.list_of_names.append(names)
            self.size = self.size + len(names)
        
        self.idx = np.zeros(self.size)
        self.simulation_time = np.zeros(self.size)
        self.applied_voltage = np.zeros(self.size)
        self.integrated_charges = np.zeros((200,self.size)) #hard coded for now
        self.polarizations = np.zeros((64,64,200,self.size))
        self.charge_map = np.zeros((64,64,200,self.size))
        self.efields = np.zeros((64,64,200,self.size))
        self.Phi_SC = np.zeros((64,64,self.size)) #min, max
        self.integrated_voltage = np.zeros(self.size)
        self.voltage_sc_hiz = np.zeros((2,self.size))
        self.switch_plts = []
    
        self.counter = 0
        for ix in range(len(self.simulation_paths)):
            simulation_path = self.simulation_paths[ix]
            names = self.list_of_names[ix]
            for filename in names:
                npfile = np.load(simulation_path+filename)
                self.idx[self.counter] = int(npfile['ts'])
                self.simulation_time[self.counter] = float(npfile['simulation_time'])
                self.applied_voltage[self.counter] = float(npfile['applied_voltage'])
                nlayers = npfile['integrated_charges'].shape[-1]
                self.integrated_charges[:nlayers,self.counter] = npfile['integrated_charges']
                self.polarizations[:,:,:nlayers,self.counter] = npfile['polarizations']
                self.charge_map[:,:,:nlayers,self.counter] = npfile['charge_map']
                self.efields[:,:,:nlayers,self.counter] = npfile['efields']
                self.integrated_voltage[self.counter] = float(npfile['integrated_voltage'])
                self.Phi_SC[:,:,self.counter] = npfile['Phi_SC']
                self.voltage_sc_hiz[:,self.counter] = npfile['voltage_sc_hiz']
                try:
                    self.switch_plts = npfile['switch_plts']
                except ValueError:
                    self.switch_plts = []
                self.counter = self.counter + 1
                
                
                
                #npfile = np.load(simulation_path+filename)
                #self.idx[self.counter] = int(npfile['ts'])
                #self.simulation_time[self.counter] = float(npfile['simulation_time'])
                #self.applied_voltage[self.counter] = float(npfile['applied_voltage'])
                #self.integrated_charge[self.counter] = float(npfile['integrated_charge'])
                #self.polarization[:, :, self.counter] = npfile['polarization']
                #self.efield[:, :, self.counter] = npfile['efield']
                #self.switch_plts = npfile['switch_plts']
                #self.counter = self.counter + 1
        
        self.switchers = []
        for switch_plt in self.switch_plts:
            counter = 0
            for _id in self.idx:
                if _id == switch_plt:
                    self.switchers.append(counter)
                    continue
                else:
                    counter = counter + 1
        #self.switchers_series = pd.Series(switchers)
        
        v_app = []
        v_fe_avg = []
        v_sc_min = []
        v_sc_max = []
        q = []
        charges_at_top = []
        for sw in self.switchers:
            top = np.nonzero(self.integrated_charges[:,-1])[0][-1]
            charges_at_top.append(self.integrated_charges[top-1,sw])
            v_app.append(self.applied_voltage[sw])
            v_fe_avg.append(self.applied_voltage[sw] - self.integrated_voltage[sw])
            #print(sw, np.round(self.applied_voltage[sw], 3), np.round(self.integrated_voltage[sw],3), np.round(self.applied_voltage[sw] - self.integrated_voltage[sw], 3))
            v_sc_min.append(self.voltage_sc_hiz[0,sw])
            v_sc_max.append(self.voltage_sc_hiz[1,sw])
            q.append(self.integrated_charges[top-1,sw])
            
        self.q_vs_v_app_series = pd.Series(q, index=v_app)
        self.q_vs_v_fe_avg_series = pd.Series(q, index=v_fe_avg)
        self.q_vs_v_sc_min_series = pd.Series(q, index=v_sc_min)
        self.q_vs_v_sc_max_series = pd.Series(q, index=v_sc_max)
        
        self.capacitance_series = self.q_vs_v_app_series.diff()/self.q_vs_v_app_series.index.to_series().diff()
        self.capacitance_series = self.capacitance_series.replace([np.inf, -np.inf], np.nan)
        self.capacitance_series = self.capacitance_series.dropna()
        
        #self.q_vs_v_fe_avg_series = self.q_vs_v_fe_avg_series.sort_index()
        self.capacitance_fe_avg_series = self.q_vs_v_fe_avg_series.diff()/self.q_vs_v_fe_avg_series.index.to_series().diff()
        self.capacitance_fe_avg_series = self.capacitance_fe_avg_series.replace([np.inf, -np.inf], np.nan)
        self.capacitance_fe_avg_series = self.capacitance_fe_avg_series.dropna()
        #self.capacitance_avg = self.capacitance_series.mean()
        
        #self.capacitance_sc_series = self.q_vs_v_sc_series.diff()/self.q_vs_v_sc_series.index.to_series().diff()
        #self.capacitance_sc_series = self.capacitance_sc_series.replace([np.inf, -np.inf], np.nan)
        #self.capacitance_sc_series = self.capacitance_sc_series.dropna()
        
        
        
        self.v_sc_max = pd.Series(v_sc_max, index=v_app)
        
    def __len__(self):
        return self.size

    def __getitem__(self, element):
        #if train and self.it in self.train_ids:
        X = self.efield[:,:,element]
        y = self.integrated_charge[element]
            
        return X, y


class patternsDB_npz_orig(Dataset):
    def __init__(self, simulations_path, *args, bins=[], normalize=True, normalization=0): # should be 1 or many simulation directories
        
        self.size = len(args)
        self.idx = np.zeros(self.size)
        self.applied_voltage = np.zeros(self.size)
        self.integrated_charge = np.zeros((50,self.size)) #hard coded for now
        self.polarization = np.zeros((64,64,50,self.size))
        self.efield = np.zeros((64,64,50,self.size))
        self.labels = []
        self.normalization = normalization
            
        self.counter = 0
        for npz in args:
            npfile = np.load(simulations_path+npz)
            self.idx[self.counter] = int(npfile['ts'])
            self.applied_voltage[self.counter] = float(npfile['applied_voltage'])
            nlayers = float(npfile['integrated_charges']).shape[-1]
            self.integrated_charges[:nlayers,self.counter] = float(npfile['integrated_charges']) 
            self.polarization[:,:,:nlayers,self.counter] = npfile['polarizations']
            self.efield[:,:,:nlayers,self.counter] = npfile['efields']
            self.counter = self.counter + 1
            
        if normalize and self.normalization == 0:
            mines = np.inf
            maxes = -np.inf
            for e in range(self.size):
                flat = self.efield[:,:,e].flatten()
                mine = flat.min()
                mines = mine if mine < mines else mines
                maxe = flat.max()
                maxes = maxe if maxe > maxes else maxes
            self.normalization = -mines if -mines > maxes else maxes
            for e in range(self.size):
                self.efield[:, :, e] = np.divide(self.efield[:, :, e], self.normalization)
        
        if normalize and self.normalization != 0:
            for e in range(self.size):
                self.efield[:, :, e] = np.divide(self.efield[:, :, e], self.normalization)
        
        self.bins = bins
        if len(self.bins) == 0:
            charges_series = pd.Series(self.integrated_charge)
            for q in range(0,101,10):
                quantile = q/100.0
                self.bins.append(charges_series.sort_values().quantile(quantile))
        
        #for e in range(self.counter):
        #    label = np.nan
        #    for b in range(len(self.bins[:-1])):
        #        high, low = self.bins[b], self.bins[b+1]
        #        if self.integrated_charge[nlayers-1,e] >= high and self.integrated_charge[nlayers-1,e] < low:
        #            #print(self.integrated_charge[e],'between', self.bins[b], 'and', self.bins[b+1])
        #            label =  str(self.bins[b+1])
        #            break

        #    self.labels.append(label)
        
    def get_normalization(self):
        return self.normalization
            
    def get_bins(self):
        return self.bins
            
    def __len__(self):
        return self.size

    def __getitem__(self, element):
        X = self.efields[:,:,nlayers-1,element]
        y = self.integrated_charges[nlayers-1,element]
        #y = self.labels[element]
            
        return X, y
        
class patternsDB_npz(Dataset):
    def __init__(self, simulations_path, *args):#, bins=[], normalize=True, normalization=0): # should be 1 or many simulation directories
        
        self.size = len(args)
        self.idx = np.zeros(self.size)
        self.applied_voltage = np.zeros(self.size)
        self.integrated_charges = np.zeros((150,self.size)) #hard coded for now
        self.polarizations = np.zeros((64,64,150,self.size))
        self.efields = np.zeros((64,64,150,self.size))
        self.Phi_SC = np.zeros((64,64,self.size)) #min, max
        self.voltage_sc_hiz = np.zeros((2,self.size))
        self.thicknesses = np.zeros((3,self.size))
            
        self.counter = 0
        for npz in args:
            npfile = np.load(simulations_path+npz)
            self.idx[self.counter] = int(npfile['ts'])
            #print(np.isclose(float(npfile['applied_voltage'], 0)))
            #if np.isclose(float(npfile['applied_voltage']), 0):
            #    continue
            self.applied_voltage[self.counter] = float(npfile['applied_voltage'])
            nlayers = npfile['integrated_charges'].shape[-1]
            self.integrated_charges[:nlayers,self.counter] = npfile['integrated_charges']
            self.polarizations[:,:,:nlayers,self.counter] = npfile['polarizations']
            self.efields[:,:,:nlayers,self.counter] = npfile['efields']
            self.voltage_sc_hiz[:,self.counter] = npfile['voltage_sc_hiz']
            self.thicknesses[:,self.counter] = npfile['thicknesses']
            self.counter = self.counter + 1
        #print(self.applied_voltage)
        self.thicknesses = torch.tensor(self.thicknesses,dtype=torch.float32)
        self.applied_voltage = torch.tensor(self.applied_voltage,dtype=torch.float32)
        self.voltage_sc_hiz = torch.tensor(self.voltage_sc_hiz,dtype=torch.float32) 
        
        #print(torch.mean(self.voltage_sc_hiz), torch.std(self.voltage_sc_hiz))
        
        self.de_thickness_mu = torch.mean(self.thicknesses[1,:]).item()
        self.de_thickness_std = torch.std(self.thicknesses[1,:]).item()
        
        self.fe_thickness_mu = torch.mean(self.thicknesses[2,:]).item()
        self.fe_thickness_std = torch.std(self.thicknesses[2,:]).item()
        
        self.applied_voltage_mu = torch.mean(self.applied_voltage[:]).item()
        self.applied_voltage_std = torch.std(self.applied_voltage[:]).item()
            
    def __len__(self):
        return self.size

    def __getitem__(self, element):
        y = self.voltage_sc_hiz[1,element] #Phi max
        #y = (self.voltage_sc_hiz[1,element] - mu) / std #Phi max
        #X = np.array([self.thicknesses[1, element], self.thicknesses[2,element], self.applied_voltage[element]])
        X = np.array([(self.thicknesses[1, element] - self.de_thickness_mu) / self.de_thickness_std,
                      (self.thicknesses[2,element] - self.fe_thickness_mu) / self.fe_thickness_std,
                      (self.applied_voltage[element] - self.applied_voltage_mu) / self.applied_voltage_std])
        #X = self.efields[:,:,nlayers-1,element]
        #y = self.integrated_charges[nlayers-1,element]
        #y = self.labels[element]
        #print(X, y)
            
        return X, y

class patternsDB_npz_abg(Dataset):
    def __init__(self, simulations_path, *args):
        
        self.size = len(args)
        self.idx = np.zeros(self.size)
        self.applied_voltage = np.zeros(self.size)
        self.integrated_charges = np.zeros((150,self.size)) #hard coded for now
        self.polarizations = np.zeros((64,64,150,self.size))
        self.efields = np.zeros((64,64,150,self.size))
        self.Phi_SC = np.zeros((64,64,self.size)) #min, max
        self.voltage_sc_hiz = np.zeros((2,self.size))
        self.thicknesses = np.zeros((3,self.size))
        self.material_properties = np.zeros((4, self.size))
            
        self.counter = 0
        for npz in args:
            npfile = np.load(simulations_path+npz)
            self.idx[self.counter] = int(npfile['ts'])
            self.applied_voltage[self.counter] = float(npfile['applied_voltage'])
            nlayers = npfile['integrated_charges'].shape[-1]
            self.integrated_charges[:nlayers,self.counter] = npfile['integrated_charges']
            self.polarizations[:,:,:nlayers,self.counter] = npfile['polarizations']
            self.efields[:,:,:nlayers,self.counter] = npfile['efields']
            self.voltage_sc_hiz[:,self.counter] = npfile['voltage_sc_hiz']
            self.thicknesses[:,self.counter] = npfile['thicknesses']
            self.material_properties[:,self.counter] = npfile['material_properties']
            self.counter = self.counter + 1
        self.thicknesses = torch.tensor(self.thicknesses,dtype=torch.float32)
        self.material_properties = torch.tensor(self.material_properties,dtype=torch.float32)
        self.applied_voltage = torch.tensor(self.applied_voltage,dtype=torch.float32)
        self.voltage_sc_hiz = torch.tensor(np.nan_to_num(self.voltage_sc_hiz),dtype=torch.float32) 
        
        self.de_thickness_mu = torch.mean(self.thicknesses[1,:]).item()
        self.de_thickness_std = torch.std(self.thicknesses[1,:]).item()
        
        self.fe_thickness_mu = torch.mean(self.thicknesses[2,:]).item()
        self.fe_thickness_std = torch.std(self.thicknesses[2,:]).item()
        
        self.alpha_mu = torch.mean(self.material_properties[0,:]).item()
        self.alpha_std = torch.std(self.material_properties[0,:]).item()
        
        self.beta_mu = torch.mean(self.material_properties[1,:]).item()
        self.beta_std = torch.std(self.material_properties[1,:]).item()
        
        self.gamma_mu = torch.mean(self.material_properties[2,:]).item()
        self.gamma_std = torch.std(self.material_properties[2,:]).item()
        
        self.g11_mu = torch.mean(self.material_properties[3,:]).item()
        self.g11_std = torch.std(self.material_properties[3,:]).item()
        
        self.applied_voltage_mu = torch.mean(self.applied_voltage[:]).item()
        self.applied_voltage_std = torch.std(self.applied_voltage[:]).item()
            
    def __len__(self):
        return self.size

    def __getitem__(self, element):
        y = self.voltage_sc_hiz[1,element] #Phi max
        X = np.array([(self.thicknesses[1, element] - self.de_thickness_mu) / self.de_thickness_std,
                      (self.thicknesses[2,element] - self.fe_thickness_mu) / self.fe_thickness_std,
                      (self.material_properties[0,element] - self.alpha_mu) / self.alpha_std,
                      (self.material_properties[1,element] - self.beta_mu) / self.beta_std,
                      (self.material_properties[2,element] - self.gamma_mu) / self.gamma_std,
                      (self.material_properties[3,element] - self.g11_mu) / self.g11_std,
                      (self.applied_voltage[element] - self.applied_voltage_mu) / self.applied_voltage_std])
            
        return X, y
    
class patternsDB_npz_abg_ncfet(Dataset):
    def __init__(self, simulations_path, *args, **kwargs):
        
        self.size = len(args)
        self.idx = np.zeros(self.size)
        self.applied_voltage = np.zeros(self.size)
        self.xs_capacitance = np.zeros(self.size)
        self.integrated_charges = np.zeros((150,self.size)) #hard coded for now
        self.polarizations = np.zeros((64,64,150,self.size))
        self.efields = np.zeros((64,64,150,self.size))
        self.Phi_SC = np.zeros((64,64,self.size)) #min, max
        self.voltage_sc_hiz = np.zeros((2,self.size))
        self.thicknesses = np.zeros((3,self.size))
        self.material_properties = np.zeros((4, self.size))
            
        self.counter = 0
        for npz in args:
            npfile = np.load(simulations_path+npz)
            self.idx[self.counter] = int(npfile['ts'])
            self.applied_voltage[self.counter] = float(npfile['applied_voltage'])
            nlayers = npfile['integrated_charges'].shape[-1]
            self.integrated_charges[:nlayers,self.counter] = npfile['integrated_charges']
            self.polarizations[:,:,:nlayers,self.counter] = npfile['polarizations']
            self.efields[:,:,:nlayers,self.counter] = npfile['efields']
            self.voltage_sc_hiz[:,self.counter] = npfile['voltage_sc_hiz']
            self.thicknesses[:,self.counter] = npfile['thicknesses']
            self.material_properties[:,self.counter] = npfile['material_properties']
            try:
                self.xs_capacitance[self.counter] = kwargs[npz]
            except KeyError:
                pass
            self.counter = self.counter + 1
                
        self.thicknesses = torch.tensor(self.thicknesses,dtype=torch.float32)
        self.material_properties = torch.tensor(self.material_properties,dtype=torch.float32)
        self.applied_voltage = torch.tensor(self.applied_voltage,dtype=torch.float32)
        self.xs_capacitance = torch.tensor(self.xs_capacitance,dtype=torch.float32)
        self.xs_capacitance = torch.nan_to_num(self.xs_capacitance)
        self.voltage_sc_hiz = torch.tensor(np.nan_to_num(self.voltage_sc_hiz),dtype=torch.float32) 
        
        self.de_thickness_mu = torch.mean(self.thicknesses[1,:]).item()
        self.de_thickness_std = torch.std(self.thicknesses[1,:]).item()
        
        self.fe_thickness_mu = torch.mean(self.thicknesses[2,:]).item()
        self.fe_thickness_std = torch.std(self.thicknesses[2,:]).item()
        
        self.alpha_mu = torch.mean(self.material_properties[0,:]).item()
        self.alpha_std = torch.std(self.material_properties[0,:]).item()
        
        self.beta_mu = torch.mean(self.material_properties[1,:]).item()
        self.beta_std = torch.std(self.material_properties[1,:]).item()
        
        self.gamma_mu = torch.mean(self.material_properties[2,:]).item()
        self.gamma_std = torch.std(self.material_properties[2,:]).item()
        
        self.g11_mu = torch.mean(self.material_properties[3,:]).item()
        self.g11_std = torch.std(self.material_properties[3,:]).item()
        
        self.applied_voltage_mu = torch.mean(self.applied_voltage[:]).item()
        self.applied_voltage_std = torch.std(self.applied_voltage[:]).item()
        
        self.xs_capacitance_mu = torch.mean(self.xs_capacitance[:]).item()
        self.xs_capacitance_std = torch.std(self.xs_capacitance[:]).item()
        
        print(self.xs_capacitance, self.xs_capacitance_mu, self.xs_capacitance_std)
            
    def __len__(self):
        return self.size

    def __getitem__(self, element):
        y = (self.xs_capacitance[element] - self.xs_capacitance_mu) / self.xs_capacitance_std
        X = np.array([(self.thicknesses[1, element] - self.de_thickness_mu) / self.de_thickness_std,
                      (self.thicknesses[2,element] - self.fe_thickness_mu) / self.fe_thickness_std,
                      (self.material_properties[0,element] - self.alpha_mu) / self.alpha_std,
                      (self.material_properties[1,element] - self.beta_mu) / self.beta_std,
                      (self.material_properties[2,element] - self.gamma_mu) / self.gamma_std,
                      (self.material_properties[3,element] - self.g11_mu) / self.g11_std,
                      (self.applied_voltage[element] - self.applied_voltage_mu) / self.applied_voltage_std])
            
        return X, y
    
class patternsDB_npz_ncfet(Dataset):
    def __init__(self, simulations_path, *args, **kwargs):
        
        self.size = len(args)
        self.idx = np.zeros(self.size)
        self.applied_voltage = np.zeros(self.size)
        self.xs_capacitance = np.zeros(self.size)
        self.integrated_charges = np.zeros((150,self.size)) #hard coded for now
        self.polarizations = np.zeros((64,64,150,self.size))
        self.efields = np.zeros((64,64,150,self.size))
        self.Phi_SC = np.zeros((64,64,self.size)) #min, max
        self.voltage_sc_hiz = np.zeros((2,self.size))
        self.thicknesses = np.zeros((3,self.size))
        self.material_properties = np.zeros((4, self.size))
        self.theta = np.zeros(self.size)
            
        self.counter = 0
        for npz in args:
            npfile = np.load(simulations_path+npz)
            self.idx[self.counter] = int(npfile['ts'])
            self.applied_voltage[self.counter] = float(npfile['applied_voltage'])
            nlayers = npfile['integrated_charges'].shape[-1]
            self.integrated_charges[:nlayers,self.counter] = npfile['integrated_charges']
            self.polarizations[:,:,:nlayers,self.counter] = npfile['polarizations']
            self.efields[:,:,:nlayers,self.counter] = npfile['efields']
            self.voltage_sc_hiz[:,self.counter] = npfile['voltage_sc_hiz']
            self.thicknesses[:,self.counter] = npfile['thicknesses']
            self.material_properties[:,self.counter] = npfile['material_properties']
            self.theta[self.counter] = np.arccos(self.material_properties[0,self.counter] / 2.5e9)
            try:
                self.xs_capacitance[self.counter] = kwargs[npz]
            except KeyError:
                pass
            self.counter = self.counter + 1
                
        self.thicknesses = torch.tensor(self.thicknesses,dtype=torch.float32)
        self.theta = torch.tensor(self.theta,dtype=torch.float32)
        self.applied_voltage = torch.tensor(self.applied_voltage,dtype=torch.float32)
        self.xs_capacitance = torch.tensor(self.xs_capacitance,dtype=torch.float32)
        self.xs_capacitance = torch.nan_to_num(self.xs_capacitance)
        self.material_properties = torch.tensor(self.material_properties,dtype=torch.float32)
        
        self.de_thickness_mu = torch.mean(self.thicknesses[1,:]).item()
        self.de_thickness_std = torch.std(self.thicknesses[1,:]).item()
        
        self.fe_thickness_mu = torch.mean(self.thicknesses[2,:]).item()
        self.fe_thickness_std = torch.std(self.thicknesses[2,:]).item()
        
        self.theta_mu = torch.mean(self.theta).item()
        self.theta_std = torch.std(self.theta).item()
        
        self.g11_mu = torch.mean(self.material_properties[3,:]).item()
        self.g11_std = torch.std(self.material_properties[3,:]).item()
        
        self.applied_voltage_mu = torch.mean(self.applied_voltage[:]).item()
        self.applied_voltage_std = torch.std(self.applied_voltage[:]).item()
        
        self.xs_capacitance_mu = torch.mean(self.xs_capacitance[:]).item()
        self.xs_capacitance_std = torch.std(self.xs_capacitance[:]).item()
            
    def __len__(self):
        return self.size

    def __getitem__(self, element):
        y = (self.xs_capacitance[element] - self.xs_capacitance_mu) / self.xs_capacitance_std
        X = np.array([(self.thicknesses[1, element] - self.de_thickness_mu) / self.de_thickness_std,
                      (self.thicknesses[2,element] - self.fe_thickness_mu) / self.fe_thickness_std,
                      (self.theta[element] - self.theta_mu) / self.theta_std,
                      (self.material_properties[3,element] - self.g11_mu) / self.g11_std,
                      (self.applied_voltage[element] - self.applied_voltage_mu) / self.applied_voltage_std])
            
        return X, y
    
class patternsDB_npz_ncfet_masked(Dataset):
    def __init__(self, simulations_path, *args, **kwargs):
        
        self.size = len(args)
        self.idx = np.zeros(self.size)
        self.applied_voltage = np.zeros(self.size)
        self.xs_capacitance = np.zeros(self.size)
        self.integrated_charges = np.zeros((150,self.size)) #hard coded for now
        self.polarizations = np.zeros((64,64,150,self.size))
        self.efields = np.zeros((64,64,150,self.size))
        self.Phi_SC = np.zeros((64,64,self.size)) #min, max
        self.voltage_sc_hiz = np.zeros((2,self.size))
        self.thicknesses = np.zeros((3,self.size))
        self.material_properties = np.zeros((4, self.size))
        self.theta = np.zeros(self.size)
            
        self.counter = 0
        self.no_nan_list = []
        self.bad_npz_list = []
        for npz in args:
            npfile = np.load(simulations_path+npz)
            self.idx[self.counter] = int(npfile['ts'])
            self.applied_voltage[self.counter] = float(npfile['applied_voltage'])
            nlayers = npfile['integrated_charges'].shape[-1]
            self.integrated_charges[:nlayers,self.counter] = npfile['integrated_charges']
            self.polarizations[:,:,:nlayers,self.counter] = npfile['polarizations']
            self.efields[:,:,:nlayers,self.counter] = npfile['efields']
            self.voltage_sc_hiz[:,self.counter] = npfile['voltage_sc_hiz']
            self.thicknesses[:,self.counter] = npfile['thicknesses']
            self.material_properties[:,self.counter] = npfile['material_properties']
            self.theta[self.counter] = np.arccos(self.material_properties[0,self.counter] / 2.5e9)
            try:
                self.xs_capacitance[self.counter] = kwargs[npz]
                self.no_nan_list.append(self.counter)
            except KeyError:
                self.xs_capacitance[self.counter] = np.nan
                self.bad_npz_list.append(npz)
            self.counter = self.counter + 1
                
        self.thicknesses = torch.tensor(self.thicknesses[:,self.no_nan_list],dtype=torch.float32)
        self.theta = torch.tensor(self.theta[self.no_nan_list],dtype=torch.float32)
        self.applied_voltage = torch.tensor(self.applied_voltage[self.no_nan_list],dtype=torch.float32)
        self.xs_capacitance = torch.tensor(self.xs_capacitance[self.no_nan_list],dtype=torch.float32)
        #self.xs_capacitance = masked_tensor(self.xs_capacitance, ~torch.isnan(self.xs_capacitance))
        #self.xs_capacitance = as_masked_tensor(self.xs_capacitance, self.mt)
        self.xs_capacitance = torch.nan_to_num(self.xs_capacitance) # Check
        self.material_properties = torch.tensor(self.material_properties[:,self.no_nan_list],dtype=torch.float32)
        
        self.de_thickness_mu = torch.mean(self.thicknesses[1,:]).item()
        self.de_thickness_std = torch.std(self.thicknesses[1,:]).item()
        
        self.fe_thickness_mu = torch.mean(self.thicknesses[2,:]).item()
        self.fe_thickness_std = torch.std(self.thicknesses[2,:]).item()
        
        self.theta_mu = torch.mean(self.theta).item()
        self.theta_std = torch.std(self.theta).item()
        
        self.g11_mu = torch.mean(self.material_properties[3,:]).item()
        self.g11_std = torch.std(self.material_properties[3,:]).item()
        
        self.applied_voltage_mu = torch.mean(self.applied_voltage[:]).item()
        self.applied_voltage_std = torch.std(self.applied_voltage[:]).item()
        
        self.xs_capacitance_mu = torch.mean(self.xs_capacitance[:]).item()
        self.xs_capacitance_std = torch.std(self.xs_capacitance[:]).item() 
            
    def __len__(self):
        return self.size

    def __getitem__(self, element):
        y = (self.xs_capacitance[element] - self.xs_capacitance_mu) / self.xs_capacitance_std
        X = np.array([(self.thicknesses[1, element] - self.de_thickness_mu) / self.de_thickness_std,
                      (self.thicknesses[2,element] - self.fe_thickness_mu) / self.fe_thickness_std,
                      (self.theta[element] - self.theta_mu) / self.theta_std,
                      (self.material_properties[3,element] - self.g11_mu) / self.g11_std,
                      (self.applied_voltage[element] - self.applied_voltage_mu) / self.applied_voltage_std])
            
        return X, y