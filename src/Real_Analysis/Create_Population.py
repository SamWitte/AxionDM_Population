import numpy as np
import os
import glob
import h5py
from scipy.special import erf, erfinv
import argparse
from scipy import stats
from scipy.integrate import solve_ivp, odeint
from fileNaming import *

parser = argparse.ArgumentParser()
parser.add_argument('--fIn',  help='Input file')
parser.add_argument('--PopN', type=int, default=1, help='Indx of population')
parser.add_argument('--Ntasks', type=int, default=1, help='Indx of population')
parser.add_argument('--Nscripts', type=int, default=1, help='Indx of population')
parser.add_argument('--gauss_approx', type=int, default=1)
parser.add_argument('--Pmin', default=1e-3)

args = parser.parse_args()

Ntasks=args.Ntasks


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
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
pop_general = True

script_dir = "scripts/"
if not os.path.exists(script_dir):
    os.mkdir(script_dir)
run_script_maker = True
PopIdx=args.PopN
num_scripts = args.Nscripts
script_tag = "_Pop_{:.0f}_".format(PopIdx)

if args.gauss_approx == 1:
    gauss_approx = True
else:
    gauss_approx = False
  
Pmin=float(args.Pmin)

ncall=3000
nbins=3
maxitrs=5
theta_err=0.04

mag_mod="Dipole"
B_DQ=0.1
ThetaQ=0.0
PhiQ=0.0
eta_fill=0.2
gagg=1.0e-12
reflect_LFL = False
delta_on_v = False
compute_point = True
return_width = True


            
    
    
def run_pulsar_population(output_dir, MassA, B0_c, sig_B0, P0_c, sig_P0, tau_ohm, ftag, gauss_approx=True, Pmin=1e-3):
        
    N_pulsars = total_num_pulsars(young=True)
    N_pulsars_old = total_num_pulsars(young=False, tau_ohm=tau_ohm)

    print("Number of pulsars: \t", N_pulsars + N_pulsars_old)
    sve_array = []

    for i in range(N_pulsars + N_pulsars_old):
        if i < N_pulsars:
            young = True
        else:
            young = False
            
        alive = False
        
        while (not alive):
            B_0 = draw_field(B0_c, sig_B0)
            if gauss_approx:
                P = draw_period(P0_c, sig_P0)
            else:
                P = draw_period_PL(P0_c, sig_P0, Pabsmin=Pmin)
                
            alive = ((B_0 / P**2)  > (0.34 * 1e12))
            
            
        if i%1000 ==0:
            print("{:d} of {:d}".format(i, N_pulsars + N_pulsars_old))
            
        chi = draw_chi()
        # position draw
        
        phi_x, theta_x, r_x = sample_NS_position(young=young) # rx in kpc
    
        rho_DM = NFW(r_x)
        vNS = v0_NS(r_x)
        vDM = v0_DM(r_x)
        
        locNS = r_x * np.array([np.sin(theta_x) * np.cos(phi_x), np.sin(theta_x) * np.sin(phi_x), np.cos(theta_x)])
        xE = np.array([0.0, 8.3, 0.0])
        dist_earth = np.sqrt(np.sum((xE - locNS)**2))

        # draw NS mass and radius
        MassNS = draw_mass() # 1201.1006
        radiusNS = draw_radius() # km [1709.05013, 1912.05705]
        view_angle = draw_chi()

        age = draw_uniform_age(young=young)  # [yr]
        # print("pre evol")
        tmes, Bfinal, Pfinal, chifinal = evolve_pulsars(B_0, P, chi, age, N_time=1000, tau_ohm=tau_ohm)
        

        # print("{:.2e}   {:.2e}   {:.3f}   {:.3f}".format(tmes[-1], Bfinal, Pfinal[-1], chifinal[-1]))
        dop_S = (vNS/2.998e5) * np.sin(sample_theta()) * np.sin(np.random.rand() * 2*np.pi)
        
        sve_array.append([i, B_0, P, chi, Bfinal, Pfinal[-1], chifinal[-1], age/1e6, rho_DM, vDM, locNS[0], locNS[1], locNS[2], MassNS, radiusNS, view_angle, dop_S])
        # idx, B_ini [G], P_ini [s], thetaM_ini [rad], B_out [G], P_out [s], thetaM_out [rad], Age [Myr], rhoDM [GeV / cm^3], v0_DM [km /s], x [kpc], y [kpc], z [kpc],  Mass NS [M_odot], radiusNS [km], viewing angle [rad], vNS [km/s]
           
        
    f_out = file_outName(output_dir, MassA, ftag, PopIdx, tau_ohm, B0_c, P0_c, sig_B0, sig_P0, return_pop=False)
        
    sve_array = np.asarray(sve_array)
    np.savetxt(f_out, sve_array, fmt='%.5e')
        
    return

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
    
    

def script_pop(num_scripts, PopIdx, script_dir, output_dir, MassA, ftag, tau_ohm, B0_c, P0_c, sig_B0, sig_P0, ncall, nbins, maxitrs, theta_err, mag_mod, B_DQ, PhiQ, ThetaQ, reflect_LFL=False, delta_on_v=True, compute_point=True, return_width=True, eta_fill=0.2,  gagg=1e-12, script_tag="_"):
    arr_text = np.empty(num_scripts, dtype=object)
    
    
    f_load = file_outName(output_dir, MassA, ftag, PopIdx, tau_ohm, B0_c, P0_c, sig_B0, sig_P0, return_pop=False)
    
    data_in = np.loadtxt(f_load)
            
    
    f_out = file_outName(output_dir, MassA, ftag, PopIdx, tau_ohm, B0_c, P0_c, sig_B0, sig_P0, return_pop=True)

    
    out_file = f_out
    # write into to each script
    
    block_text = "srun -n 1 --exclusive --mem=$memPerjob --cpus-per-task=1 julia --threads 1 Gen_Vegas_Server.jl --MassA $MassA --B0 $B0 --ThetaM $ThetaM --rotW $rotW --AxG $gagg --Mass_NS $Mass_NS --rNS $rNS  --ftag $ftag --add_tau true --Thick true --thetaN 1 --theta_target $thetaV --rhoDM $rhoDM --vmean_ax $vmean_ax --debug false --ncalls $ncall --nbins $nbins --maxitrs $maxitrs --theta_err $theta_err --null_fill $null_fill --reflect_LFL $reflect_LFL --delta_on_v $delta_on_v --compute_point $compute_point --return_width $return_width --output_dir $out_file & \n"
   
    
    # SERVER SPECIFIC -- writing for Hydra
    for i in range(num_scripts):
        arr_text[i] = "#!/bin/bash -l \ncd ../../RayTracing/ \nmodule load julia \n"
        arr_text[i] += "declare -i memPerjob \nmemPerjob=$((SLURM_MEM_PER_CPU/SLURM_NTASKS)) \n\n"
        
        arr_text[i] += "out_file=../Real_Analysis/"+out_file+"\n"
        arr_text[i] += "ncall={:.0f}\n".format(ncall)
        arr_text[i] += "nbins={:.0f}\n".format(nbins)
        arr_text[i] += "maxitrs={:.0f}\n".format(maxitrs)
        arr_text[i] += "theta_err={:.2f}\n".format(theta_err)
        
        
        
        arr_text[i] += "MassA={:.3e}\n".format(MassA)
        arr_text[i] += "gagg={:.3e}\n".format(gagg)
        
        
        arr_text[i] += "null_fill={:.2f}\n".format(eta_fill)
        arr_text[i] += "mag_mod="+mag_mod+"\n"
        arr_text[i] += "B_DQ={:.3f}\n".format(B_DQ)
        arr_text[i] += "PhiQ={:.3f}\n".format(PhiQ)
        arr_text[i] += "ThetaQ={:.3f}\n".format(ThetaQ)
        if reflect_LFL:
            arr_text[i] += "reflect_LFL=true\n"
        else:
            arr_text[i] += "reflect_LFL=false\n"
        if delta_on_v:
            arr_text[i] += "delta_on_v=true\n"
        else:
            arr_text[i] += "delta_on_v=false\n"
        if compute_point:
            arr_text[i] += "compute_point=true\n"
        else:
            arr_text[i] += "compute_point=false\n"
        if return_width:
            arr_text[i] += "return_width=true\n\n\n"
        else:
            arr_text[i] += "return_width=false\n\n\n"
        
    
    indx = 0
    for i in range(len(data_in[:,0])):
        # [i, B_0, P, chi, Bfinal[-1], Pfinal[-1], chifinal[-1], age/1e6, rho_DM, v0_DM, locNS[0], locNS[1], locNS[2], vNS[0], vNS[1], vNS[2], MassNS, radiusNS, view_angle]
        
        if check_conversion(MassA, data_in[i, 4], data_in[i, 5], data_in[i, 6]) == 0:
            continue
            
        if data_in[i, 4] > 4.4e13:
            continue
        
        newL = "B0={:.3e} \n".format(data_in[i, 4])
        newL += "rotW={:.4f} \n".format(2*np.pi / data_in[i, 5])
        newL += "ThetaM={:.4f} \n".format(data_in[i, 6])
        newL += "Mass_NS={:.3f} \n".format(data_in[i, 13])
        newL += "rNS={:.3f} \n".format(data_in[i, 14])
        newL += "thetaV={:.3f} \n".format(data_in[i, 15])
        newL += "rhoDM={:.3f} \n".format(data_in[i, 8])
        newL += "vmean_ax={:.3f} \n".format(data_in[i, 9])
        newL += "ftag=NS_{:.0f}_ \n".format(i)
        
        arr_text[indx%num_scripts] += newL + block_text
        arr_text[indx%num_scripts] += "sleep 10\n"
        indx += 1
        if indx%(Ntasks*num_scripts) == 0:
            for j in range(num_scripts):
                arr_text[j] += "wait \n\n"

    print("Number of pulsars that need to be run \t ", indx)
    for i in range(num_scripts):
        text_file = open(script_dir + "/Script_Run_"+script_tag+"RT_{:.0f}.sh".format(i), "w")
        arr_text[i] += "wait \n"
        text_file.write(arr_text[i])
        text_file.close()

    return



def RHS(t, y, beta=6e-40, tau_ohm=10.0e6, B_0=1.0e12):
    # Initial values
    
    # logB = y[0]
    B = B_0 * np.exp(- t / tau_ohm)
    P = y[0]
    chi = y[1]
    
    # check death line
    alive = ((B / P**2)  > (0.34 * 1e12))

    # Bfield Evolution
    T_core = 1e9 * (1 + t)**(-1./6.)
    # tau_Ohm = tau_ohm * 1e6
    # tau_Hall = 5e5 * (1e14 /B)

    # dLogBdT = -1.0/tau_Ohm
    # dLogBdT = -1.0/tau_Ohm

    # Period Evolution
    dPdT = beta * B**2 / P * (alive * 1.0 + 1.0 * np.sin(chi)**2)

    # Chi Evolution
    dChidT = -1.0 * beta * B**2 / P**2 * np.abs(np.sin(chi) * np.cos(chi))
    
#    return [dLogBdT ,  3.1536e+7*dPdT, 3.1536e+7*dChidT]  # 1/ yr
    return [3.1536e+7*dPdT, 3.1536e+7*dChidT]  # 1/ yr
    
def evolve_pulsars(B_0, P, chi, age, N_time=1e5, tau_ohm=10.0e6):
    
    times = np.geomspace(1, age, int(N_time))

    y0=[P, chi]
    # print("In: ", [(B_0) / 1e12, P, chi], times)
    def call_F(t, y):
        return  RHS(t, y, beta=6e-40, tau_ohm=tau_ohm, B_0=B_0)
        
    sol_tot = solve_ivp(call_F, [times[0], times[-1]], y0, t_eval=times, max_step=10000.0)
    # sol_tot = odeint(RHS, y0, times, args=(6e-40, tau_ohm, ), max_step=)
    sol = np.asarray(sol_tot.y)
    
    # sol[0,:] = np.exp(sol[0,:])
    Btoday = B_0 * np.exp(-times[-1] / tau_ohm)
    # print("Out: ", sol[:, -1])
    return times, Btoday, sol[0,:], sol[1,:]
        

        
def draw_field(B_c, sB):
    # return 10**stats.norm.rvs(loc=self.mu_B, scale=self.sigma_B)
    return 10**(B_c + np.sqrt(2) * sB * erfinv(2 * np.random.rand() - 1.0))

def draw_period(P_c, sP):
    val = stats.norm.rvs(loc=P_c, scale=sP)
    if val <= 0:
        return draw_period(P_c, sP)
    else:
        return val
        
def draw_period_PL(Pbeta, Pmax, Pabsmin=1e-3):
    val = ((-Pmax**(1 + Pbeta) + Pabsmin**(1 + Pbeta)) * (- Pabsmin**(1 + Pbeta) / (Pmax**(1 + Pbeta) - Pabsmin**(1 + Pbeta)) - np.random.rand()))**(1.0 / (1.0 + Pbeta))
    return val
        
def draw_mass():
    val = stats.norm.rvs(loc=1.28, scale=0.24)
    if val <= 0.1 or val >= 2.0:
        return draw_mass()
    else:
        return val
        
def draw_radius():
    val = stats.norm.rvs(loc=12.0, scale=0.8)
    return val

def check_conversion(MassA, B, P, thetaM):
    ne_max = 2 * (2*np.pi / P) * (B * np.cos(thetaM) * 1.95e-2) / 0.3 * 6.58e-16
    max_omegaP = np.sqrt(4*np.pi / 137 * np.abs(ne_max) / 5.00e5)
    if max_omegaP < MassA:
        return 0 # no conversion possible
    else:
        return 1 # conversion possible
        
        
def draw_chi():
    # return np.arccos(stats.uniform.rvs(loc = -1, scale = 2))
    return np.arccos(1.0 - 2.0 * np.random.rand())

def draw_uniform_age(young=True, tau_ohm=10.0e6):
    if young:
        return np.random.rand() * 30.0e6
    else:
        if tau_ohm < 1e9:
            max_age = 1e9
        else:
            max_age = 10e9
        age_good = False
        while not age_good:
            age = np.random.rand() * max_age
            if age > 30.0e6:
                age_good = True
        return age
        
        
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



        

if pop_general:
    # Makes initial input that is then used to generate server run script....
    run_pulsar_population(output_dir, MassA, B0_c, sig_B0, P0_c, sig_P0, tau_ohm, ftag, gauss_approx=gauss_approx, Pmin=args.Pmin)
    
if run_script_maker:
    # Makes script that can then be used to run Vegas
    script_pop(num_scripts, PopIdx, script_dir, output_dir, MassA, ftag, tau_ohm, B0_c, P0_c, sig_B0, sig_P0, ncall, nbins, maxitrs, theta_err, mag_mod, B_DQ, PhiQ, ThetaQ, reflect_LFL=reflect_LFL, delta_on_v=delta_on_v, compute_point=compute_point, return_width=return_width, eta_fill=eta_fill, gagg=gagg, script_tag=script_tag)

