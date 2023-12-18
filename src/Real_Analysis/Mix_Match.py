import numpy as np
import os
import glob
import h5py
import argparse
from fileNaming import *
import warnings




parser = argparse.ArgumentParser()
parser.add_argument('--fIn',  help='Input file')
args = parser.parse_args()


input_info = []
with open(args.fIn) as f:
    for line in f:
        line = line.partition('#')[0]
        line = line.rstrip()
        input_info.append(line)

MassA = float(input_info[0])
B0_c = float(input_info[1])
sig_B0 = float(input_info[2])
P0_c = float(input_info[3])
sig_P0 = float(input_info[4])
tau_ohm = float(input_info[5])
ftag = input_info[6]
output_dir = "Output_Files/"

f_out, massD = file_outName(output_dir, MassA, ftag, 1, tau_ohm, B0_c, P0_c, sig_B0, sig_P0, return_top_level=True)
# print(f_out)

def gen_population(f_out, massD, tau_ohm, MassA):
    Pop_topL = glob.glob(output_dir + f_out + "Population_*.txt")
    Npop = len(Pop_topL)
    print("Number of populations used: \t", Npop)
    generalF = np.empty(Npop, dtype=object)
    for i in range(Npop):
        generalF[i] = np.loadtxt(Pop_topL[i])
        

    N_pulsars_young = total_num_pulsars(young=True)
    N_pulsars_old = total_num_pulsars(young=False, tau_ohm=tau_ohm)
    Ntot = np.random.poisson(lam=(N_pulsars_young + N_pulsars_old))

    output_pop = []
    # New population will be re-sampled / re-weighted version of the independent realizations
    # only position is re-sampled!
    for i in range(Ntot):
        indxP = np.random.randint(Npop)
        total_entries = len(generalF[indxP])
        NS_num = np.random.randint(total_entries)
        NS_info = generalF[indxP][NS_num, :]
        
        NS_outFile = glob.glob(output_dir + f_out + massD + "Pop_{:.0f}/NS_{:.0f}__Theta*.txt".format(indxP, NS_num))
        if len(NS_outFile) == 0:
            continue
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data_in = np.loadtxt(NS_outFile[0], ndmin=2)
            if len(data_in) == 0:
                continue
        
        
        if NS_info[7] < 30.0:
            young = True
        else:
            young = False
        
        
        phi_x, theta_x, r_x = sample_NS_position(young=young) # rx in kpc
        
        vNS = v0_NS(r_x)
        dop_S = (vNS/2.998e5) * np.sin(sample_theta()) * np.sin(np.random.rand() * 2*np.pi)
        
        locNS = r_x * np.array([np.sin(theta_x) * np.cos(phi_x), np.sin(theta_x) * np.sin(phi_x), np.cos(theta_x)])
        xE = np.array([0.0, 8.3, 0.0])
        dist_earth = np.sqrt(np.sum((xE - locNS)**2))
        
        weight_rescale = (NFW(r_x) / NS_info[8]) * (NS_info[9] / v0_DM(r_x))
        
        flux_weight = data_in[:, 0] * (1.60e-12) * weight_rescale / dist_earth**2 * (3.24e-22)**2 / 1e-23 # Jy-Hz
        erg_list = -data_in[:, 1] * (1 + dop_S)
    
        for j in range(len(erg_list)):
            output_pop.append([i, flux_weight[j], erg_list[j], data_in[j, 2]])
        
    output_pop = np.asarray(output_pop)
    print("Rough estimate total radio flux @ g=1e-12 in Jy: \t ", np.sum(output_pop[:,1]) / (MassA * 1e-5 / 6.58e-16))
    return np.asarray(output_pop)

def total_num_pulsars(young=True, tau_ohm=10.0e6):
    # max_age [Myr]
    if young:
        total_num = 5.4e-5 * 30.0e6 # assume rate of 5.4e-5 / year, and consider "young population" only for 30 Myr
    else:
        if tau_ohm < 1e9:
            total_num = 345050 * ((tau_ohm * 10.0 - 30.0e6) / 10.0e9)
        else:
            total_num = 345050
        
    return np.random.poisson(int(total_num))

def nNS_dist(r): # normalized
    return r**0.07 * np.exp(-r / 0.5) * 0.468
    
def nNS_dist_old(r): # normalized
    if r < 2:
        return 5.98e3 * r**(-1.7) / 40742.2 # Num / pc^3, r in pc
    if r > 2:
        return 2.08e4 * r**(-3.5) / 40742.2


def sample_NS_position(young=True):
    phi = 2 * np.pi * np.random.rand()
    theta = sample_theta()
    
#    gam = 0.4
#    eta = 3 - gam
#    rB = 0.824 * 1e-3 # kpc, taken from arXiv:0902.3892
    if young:
        sucs = False
        while not sucs:
            u = np.random.rand()
            vv = np.random.rand()
            rr = -0.5 * np.log((1 - vv))
            frr = nNS_dist(rr) # real dist eval at rr
            grr = np.exp(- rr / 0.5) # M = 2.5
            if u < frr / (2.5 * grr):
                sucs = True
                
    else:
        u = np.random.rand()
        rr = 3.966 * u ** (10/13)
        if rr > 2:
            rr = 2.27391 / (2.1811 - 2.95371 * u + u**2)

    return phi, theta, rr / 1e3 #NS radius in kpc
    
def sample_theta():
    rand_vs = np.random.rand()
    return np.arccos(1.0 - 2.0 * rand_vs)  # rads


def NFW(r):
    rho0 = 0.346 # GeV / cm^3
    rs = 20.0 # kpc
    return rho0 / ((r / rs) * (1 + r / rs)**2) # GeV / cm^3
    
def v0_NS(r):
    # r in kpc, v0 in km/s
    return 83.74 * np.sqrt(1 / (r * 1e3)) # derived from Bens paper with gamma = 1.93
    
def v0_DM(r):
    # r in kpc, v0 in km/s
    return 122.67 * np.sqrt(1 / (r * 1e3)) # derived from Bens paper with gamma = 1


gen_population(f_out, massD, tau_ohm, MassA)
