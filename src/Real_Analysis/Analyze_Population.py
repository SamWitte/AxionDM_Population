import numpy as np
import os
import glob
import h5py
import argparse
from fileNaming import *
import warnings

parser = argparse.ArgumentParser()
parser.add_argument('--fIn',  help='Input file')
parser.add_argument('--PopN', type=int, default=1, help='Indx of population')
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

PopIdx=args.PopN
### read general file

f_out = file_outName(output_dir, MassA, ftag, PopIdx, tau_ohm, B0_c, P0_c, sig_B0, sig_P0, return_pop=False)
    
generalF = np.loadtxt(f_out)



f_out_P = file_outName(output_dir, MassA, ftag, PopIdx, tau_ohm, B0_c, P0_c, sig_B0, sig_P0, return_pop=True)

if os.path.exists(f_out_P + "/Combined_Flux.dat"):
    os.remove(f_out_P + "/Combined_Flux.dat")

all_files = glob.glob(f_out_P + "/*")

def gauss_line_approx(erg, erg_c, erg_w, cutoff=1e-4):
    val = np.exp(- (erg - erg_c)**2 / (2 * erg_w**2)) / np.sqrt(2*np.pi*erg_w**2)
    if val < cutoff:
        return 0.0
    else:
        return val
 
def sample_theta():
    rand_vs = np.random.rand()
    return np.arccos(1.0 - 2.0 * rand_vs)  # rads

### cycle through files and collect info
erg_max_shift = 1.0e-2
num_pts = 10000
erg_list = np.linspace(MassA * (1.0 - erg_max_shift), MassA * (1.0 + erg_max_shift), num_pts)
fluxV = np.zeros_like(erg_list)

hold_info = []
for i in range(len(all_files)):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fileIn = np.loadtxt(all_files[i], ndmin=2)
    if len(fileIn) == 0:
        continue
    indxP = int(all_files[i][all_files[i].find("/NS_") + 4: all_files[i].find("__Theta")])
    
    dop_S = generalF[indxP, -1]
    # dop_S = (vNS/2.998e5) * np.sin(sample_theta()) * np.sin(np.random.rand() * 2*np.pi)
    locNS = np.array([generalF[indxP, 10], generalF[indxP, 11], generalF[indxP, 12]])
    xE = np.array([0.0, 8.3, 0.0])
    dist_earth = np.sqrt(np.sum((xE - locNS)**2))
        
    flux_weight = fileIn[:, 0] * (1.60e-12) / dist_earth**2 * (3.24e-22)**2 / 1e-23 # Jy-Hz
    erg_list = -fileIn[:, 1] * (1 + dop_S)
    # hold_info.append([indxP, flux, erg_central, fileIn[3] * erg_central, probability @ g = 1e-12 1/GeV])
    for j in range(len(erg_list)):
        hold_info.append([indxP, flux_weight[j], erg_list[j], fileIn[j, 2]])
    
hold_info = np.asarray(hold_info)
# for i in range(num_pts):
#    for j in range(len(all_files)):
#        fluxV[i] += hold_info[j, 1] * gauss_line_approx(erg_list[i], hold_info[j, 2], hold_info[j, 3])

np.savetxt(f_out_P + "/Combined_Flux.dat", hold_info)

print("Rough estimate total radio flux @ g=1e-12 in Jy: \t ", np.sum(hold_info[:,1]) / (MassA * 1e-5 / 6.58e-16))


    
    
    
    
