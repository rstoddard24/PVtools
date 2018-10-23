#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 12:06:40 2018

@author: ryanstoddard
"""

import numpy as np
import pandas as pd
import math
import scipy
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from os import listdir

#Constants
pi = math.pi
heV = 4.14e-15 #eV*s
c = 2.99792e8 #m/s
kbeV = 8.6173e-5 #eV/K
keV = 8.6173e-5 #eV/K
h = 6.626e-34
kb = 1.38065e-23
q = 1.60218e-19
k = 1.3806488e-23
T = 300

CellAreacm = 0.0453
CellArea = CellAreacm*10**-4 #m^2
Ps=100 #mW/cm^2


default_figsize = mpl.rcParamsDefault['figure.figsize']
mpl.rcParams['figure.figsize'] = [1.5*val for val in default_figsize]
font = {'family' : 'DejaVu Sans',
        'weight' : 'bold',
        'size'   : 22}

mpl.rc('font', **font)
mpl.rc('axes', linewidth=3)


Directory = '../../data/JVdata/2018_10-1JV/'
names = listdir(Directory)
names_fs = []
names_rs = []
names_hold = []

#%%
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def reject_outliers(data, m):
    if len(data) > 2:
        return data[abs(data - np.mean(data)) < m * np.std(data)]
    else:
        return data

#sort names
for name in names:
    if 'liv2' in name:
        names_fs.append(name)
    elif 'liv1' in name:
        names_rs.append(name)
    elif ('hold' in name) and not ('.png' in name):
        names_hold.append(name)
        
dev_types = np.zeros(len(names_rs))
PCEs = np.zeros(len(names_rs))
Vocs = np.zeros(len(names_rs))
Jscs = np.zeros(len(names_rs))
FFs = np.zeros(len(names_rs))
hyst_idx = np.zeros(len(names_rs))


basename = 'Stoddard_2018_10-2JV_'

#%%
for ii in range(len(names_rs)):
    k = 1
    if is_number(names_rs[ii][len(basename)+3]):
        k += 1
    dev_type = float(names_rs[ii][len(basename):len(basename)+k])
    dev_types[ii] = dev_type
    
    Ldata = pd.read_csv(Directory + names_rs[ii], delimiter='\t', header=None)
    idx_end = Ldata[Ldata.iloc[:,0] == 'Jsc:'].index[0]
    Ldata = Ldata.iloc[:idx_end-1,:]
    Ldata.iloc[:,0] = pd.to_numeric(Ldata.iloc[:,0])
    Ldata.iloc[:,0]
    Ldata = np.array(Ldata)

    Ldata = np.insert(Ldata, 2, -Ldata[:,1], axis=1)

    JVinterp = interp1d(Ldata[:,0], Ldata[:,2], kind='cubic', bounds_error=False, fill_value='extrapolate')

    JscL = -JVinterp(0)
    VocL = scipy.optimize.fsolve(JVinterp,.95*max(Ldata[:,0]))
    PPV = scipy.optimize.fmin(lambda x: x*JVinterp(x),.8*VocL,disp=False)
    PCE = -PPV*JVinterp(PPV)
    FF = PCE/(JscL*VocL)*100
    
    PCEs[ii] = PCE
    Vocs[ii] = VocL
    Jscs[ii] = JscL
    FFs[ii] = FF

#Delete shunted devices / bad data
k = 0
while k < len(PCEs):
    if (Jscs[k] < 1) or (Vocs[k] < 0.5) or (FFs[k] < 25) or (FFs[k] > 85) or (PCEs[k] < 1):
        PCEs = np.delete(PCEs,k)
        Vocs = np.delete(Vocs,k)
        Jscs = np.delete(Jscs,k)
        FFs = np.delete(FFs,k)
        dev_types = np.delete(dev_types,k)
        names_rs = np.delete(names_rs,k)
    else:
        k += 1
        
#%% Get stabilized data from hold files
run_num = 16
PCE_stab = np.zeros(run_num)
ddt_PCE = np.zeros(run_num)

for ii in range(len(names_hold)):
    k = 1
    if is_number(names_hold[ii][len(basename)+3]):
        k += 1
    dev_type_hold = float(names_hold[ii][len(basename):len(basename)+k])
    
    
    hold_data =  np.genfromtxt(Directory + names_hold[ii],skip_header=1,skip_footer=4)
    
    PCE_stab[int(dev_type_hold)-1] = hold_data[-1,2]
    ddt_PCE[int(dev_type_hold)-1] = (hold_data[-1,2]-hold_data[6,2])/(hold_data[-1,0]-hold_data[6,0])

#%%

PCE_list = []
Voc_list = []
Jsc_list = []
FF_list = []

data_out = np.zeros([run_num,7])
for ii in range(run_num):
    idxs = np.argwhere(dev_types == ii +1)
    PCE_list.append(reject_outliers(PCEs[idxs],2))
    Voc_list.append(reject_outliers(Vocs[idxs],2))
    Jsc_list.append(reject_outliers(Jscs[idxs],2))
    FF_list.append(reject_outliers(FFs[idxs],2))
    data_out[ii,0] = np.mean(PCE_list[ii])
    data_out[ii,1] = np.mean(Voc_list[ii])
    data_out[ii,2] = np.mean(Jsc_list[ii])
    data_out[ii,3] = np.mean(FF_list[ii])
    
data_out[:,4] = PCE_stab
data_out[:,5] = ddt_PCE
data_out[:,6] = np.abs(ddt_PCE)
np.savetxt(Directory + 'data_to_DOE.txt',data_out)