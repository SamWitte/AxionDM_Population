import numpy as np
import os
from Interpolate_Bank import *

file_in = np.loadtxt("test.txt")
MassA = 1.0e-5
dir_out = "population_output"
file_out = "Pop_test.txt"

def run_population(MassA, file_in, dir_out, file_out, NS_population='Young'):
    full_radio = None

    for i in range(len(file_in)):
        # Desired File format
        # indx, B0, P0, Theta_View
        ThetaV = file_in[i, 3]
        Bv = file_in[i, 1]
        Pv = file_in[i, 2]
        
        phi_x, theta_x, r_x = sample_NS_position(NS_population)
        
        rho_DM = NFW(r_x)
        vNS = v0_NS(r_x)
        vDM = v0_DM(r_x)
        
        locNS = r_x * np.array([np.sin(theta_x) * np.cos(phi_x), np.sin(theta_x) * np.sin(phi_x), np.cos(theta_x)])
        xE = np.array([0.0, 8.3, 0.0])
        dist_earth = np.sqrt(np.sum((xE - locNS)**2))
    
        out = Pulsar_signal(MassA, ThetaV, Bv, Pv, gagg=1.0e-12, eps_theta=0.03, dopplerS=(vNS/2.998e5), density_rescale=rho_DM, v0_rescale=vDM)
        if np.all(out == 0):
            continue
        out[:, 1] *= (1.60e-12) / dist_earth**2 * (3.24e-22)**2 / 1e-23 # Jy-Hz (ie divide by bandwidth in Hz to get flux density)
        if full_radio == None:
            full_radio = out
        else:
            full_radio = np.vstack((full_radio, out))
    
    if not os.path.isdir(dir_out):
        os.mkdir(dir_out)
    np.savetxt(dir_out + "/" + file_out, full_radio)
    return


def nNS_dist(r): # normalized
    return r**0.07 * np.exp(-r / 0.5) * 0.468
    
def nNS_dist_old(r): # normalized
    if r < 2:
        return 5.98e3 * r**(-1.7) / 40742.2 # Num / pc^3, r in pc
    if r > 2:
        return 2.08e4 * r**(-3.5) / 40742.2


def sample_NS_position(NS_population):
    phi = 2 * np.pi * np.random.rand()
    theta = sample_theta()
    
#    gam = 0.4
#    eta = 3 - gam
#    rB = 0.824 * 1e-3 # kpc, taken from arXiv:0902.3892
    if (NS_population == 'Young') or (NS_population == 'Magnetar'):
        sucs = False
        while not sucs:
            u = np.random.rand()
            vv = np.random.rand()
            rr = -0.5 * np.log((1 - vv))
            frr = nNS_dist(rr) # real dist eval at rr
            grr = np.exp(- rr / 0.5) # M = 2.5
            if u < frr / (2.5 * grr):
                sucs = True
                
    elif NS_population == 'Old' or NS_population == 'Old_Ben':
        u = np.random.rand()
        rr = 3.966 * u ** (10/13)
        if rr > 2:
            rr = 2.27391 / (2.1811 - 2.95371 * u + u**2)
        
    else:
        print('Problem. NS population not implimented....')
        return
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



run_population(MassA, file_in, dir_out, file_out, NS_population='Young')
