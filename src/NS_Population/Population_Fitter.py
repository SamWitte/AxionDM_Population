import numpy as np
from astropy import units as u
import pygedm
from scipy import stats
from scipy.integrate import solve_ivp
import random
from scipy.special import erfinv, erf
from numba import jit
import time
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
import torch
import normflows as nf
from matplotlib import pyplot as plt
from tqdm import tqdm
import emcee
from core_test import NormalizingFlow_2
import logging;
import corner
logging.disable(logging.WARNING)

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
#################
# random.seed(10)
run_magnetars = False

NS_formationrate = 2.0 # per century
max_T = 100e6 # only simulate pulsars with ages < max_T

filePulsars = np.loadtxt("psrcat_tar/store_out/Pulsar_Population.txt")
fileMagnetars = np.loadtxt("psrcat_tar/store_out/Magnetar_Population.txt")
if not run_magnetars:
    true_pop = filePulsars[:, 1:3]
    B_max = 4.4e13
    B_min = 1e10
else:
    true_pop = np.vstack((filePulsars[:, 1:3], fileMagnetars))
    B_max = 1e15
    B_min = 1e10
    
# dont include spinning up
true_pop = true_pop[true_pop[:,1] > 0]
# refine range of B
B_true = 1e12 * np.sqrt((true_pop[:,1] / 1e-15)  * true_pop[:,0])
true_pop = true_pop[np.all(np.column_stack((B_true > B_min, B_true < B_max)), axis=1)]

# ohmic decay timescale
tau_ohmic = 10.0e6 # yrs

num_pulsars = NS_formationrate / 1e2 * max_T
print("Estimated number of pulsars in formed in last {:.2e} years: {:.2e}".format(max_T, num_pulsars))

N_samples = 1e5



@jit()
def beaming_cut(P):
    f_per = (9 * np.log10(P / 10)**2 + 3.0) * 1e-2
    # print("fper \t", f_per)
    is_pointing = np.random.choice([0, 1], p=[1.0 - f_per, f_per])
    return is_pointing # 0 = not aligned, 1 = we can see

@jit()
def Lum(P, Pdot, dist, alpha=0.48, muLcorr=0.0, sigLcorr=0.8):
    L0 = 5.69e6 # mJy * kpc^2 -- arb norm
    L1Ghz = L0 * 10**np.random.normal(loc=muLcorr, scale=sigLcorr) * (Pdot / P**3)**alpha

    return L1Ghz, L1Ghz / dist**2 # dist in kpc, flux density (second quantity in mJy)

@jit()
def sample_location(age, diskH=0.5, diskR=10.0):
    # Velocity fit from neutron star kicks paper, 2 componenent maxwellian
    # disk params in kpc
    hh = np.random.uniform(-diskH,diskH)
    rr = diskR * np.sqrt(np.random.rand())
    phi_loc =  2*np.pi * np.random.rand()
    x_cart = np.array([rr * np.cos(phi_loc), rr * np.sin(phi_loc), hh])
    
    Vel1d = 0.9 * (2*np.sqrt(190.0) * np.sqrt(np.log(1 / (1 - np.random.rand())))) + 0.1 * (2*np.sqrt(786.0) * np.sqrt(np.log(1 / (1 - np.random.rand())))) 
    theta = np.arccos(1.0 - 2.0 * random.random())
    phi = 2 * np.pi * np.random.rand()
    vel_vec = Vel1d * np.array([ np.cos(phi) * np.sin(theta), np.sin(theta) * np.sin(phi), np.cos(theta)])

    # ADD GRAV POTENTIAL OF MILKY WAY
    xpos = x_cart + vel_vec * age * (3.24078e-17 * 3.15e7) 

    return xpos # cartesian in kpc

@jit()
def min_flux_test1(P, tau_sc, threshold=0.1):
    if tau_sc / P > threshold:
        return 0
    else:
        return 1

@jit()
def min_flux_test2(P, DM, beta=2, tau=1.0e-2, DM0=60, DM1=7.5, G=0.64, Trec=21, d_freq=10e6, tobs=200, SN=10, Tsky=1.0):
    wid = 0.05 * P
    # print("Dispersion Measures \t", DM, DM0, DM1)
    Smin = SN * beta * (Trec + Tsky) / (G * 2 * d_freq * tobs) * 1e3 # mJy 
    w_e = np.sqrt(wid**2 + (beta * tau)**2 + (tau * DM / DM0)**2 + (tau * (DM - DM1) / DM1)**2 )
    return Smin * np.sqrt(w_e / (P - w_e))
    
    
@jit()
def evolve_pulsar(B0, P0, Theta_in, age, n_times=1e4, beta=6e-40, tau_Ohm=10.0e6):
    times = np.geomspace(1, age, int(n_times))
    y0=[np.log(B0), P0, Theta_in]
    
    sol_tot = solve_ivp(RHS, [times[0], times[-1]], y0, t_eval=times, args=(beta, tau_Ohm, ), max_step=500000.0)
    sol = np.asarray(sol_tot.y)
    
    Bf = np.exp(sol[0, -1])
    Pf = sol[1,-1]
    Theta_f = sol[2,-1]
    Pdot = beta * Bf**2 / Pf * (1.0 + 1.0 * np.sin(Theta_f)**2)
    
    return Bf, Pf, Theta_f, Pdot

@jit()
def RHS(t, y, beta=6e-40, tau_Ohm=10.0e6):
    
    logB = y[0]
    B = np.exp(logB)
    if B < 1e9:
        return [0.0, 0.0, 0.0]
    P = y[1]
    chi = y[2]
    # check death line
    alive = ((B / P**2)  > (0.34 * 1e12))
    # Bfield Evolution
    # T_core = 1e9 * (1 + t)**(-1./6.)
    # tau_Ohm = self.tau_ohm * 1e6
    # tau_Hall = 5e5 * (1e14 /B)

    # dLogBdT = -1.0/tau_Ohm - 1.0/tau_Hall
    dLogBdT = -1.0/tau_Ohm

    # Period Evolution
    dPdT = beta * B**2 / P * (alive * 1.0 + 1.0 * np.sin(chi)**2)

    # Chi Evolution
    dChidT = -1.0 * beta * B**2 / P**2 * np.sin(chi) * np.cos(chi)
    
    return [dLogBdT ,  3.1536e+7*dPdT, 3.1536e+7*dChidT]  # 1/ yr

@jit()
def draw_Bfield_lognorm():
    mu_B = 12.95
    sig_B = 0.55
    return 10**(mu_B + np.sqrt(2) * sig_B * erfinv(2 * random.random() - 1.0))

@jit()
def draw_period_norm():
    mu_P = 0.3
    sig_P = 0.15
    val =  mu_P + np.sqrt(2) * sig_P * erfinv(2 * random.random() - 1.0) #stats.norm.rvs(loc=mu_P, scale=sig_P)
    if val <= 0:
        return draw_period_norm()
    else:
        return val
        
@jit()
def draw_chi(Nsamps=1):
    return np.arccos(1.0 - 2.0 * np.random.random(Nsamps))
    
@jit()
def simulate_pop(num_pulsars, ages, beta=6e-40, tau_Ohm=10.0e6, width_threshold=0.1, pulsar_data=None):
    final_list = []
    
    
    for i in range(len(ages)):
    
#        if i % 1e4 == 0:
#            print("Currently at ", i, " of ", num_pulsars)
        # sample
        Theta_in = draw_chi()
        
        if type(pulsar_data) == type(None):
            B0 = draw_Bfield_lognorm()
            P0 = draw_period_norm()
        else:
            B0 = pulsar_data[i, 0]
            P0 = pulsar_data[i, 1]
        
        # evolve
        Bf, Pf, ThetaF, Pdot = evolve_pulsar(B0, P0, Theta_in, ages[i], beta=beta, tau_Ohm=tau_Ohm)
        # print(B0, Bf, P0, Pf, Theta_in, ThetaF)
        
        # is dead?
        Bdeath = 0.34 * 1e12 * (Pf)**2
        if Bf <= Bdeath:
            # print("dead")
            continue
        
        # get current location today
        xfin = sample_location(ages[i], diskH=0.5, diskR=10.0)
        if np.abs(xfin[2] > 0.5):
            # print("does not live in disk... ")
            continue
        
        xE = np.array([0.0, 8.3, 0.0])
        xfin_shift = xfin - xE
       
        dist_earth = np.sqrt(np.sum(xfin_shift**2))
        # print("Distance \t ", dist_earth)
        # compute DM and tau based on location
        GC_b = np.arcsin(xfin[2] / dist_earth)
        GC_l = np.arctan2(xfin_shift[1], xfin_shift[2])
        DM, tau_sc = pygedm.dist_to_dm(GC_l, GC_b, dist_earth * 1e3 * u.pc, method='ymw16')
        
        if Bf < 4.4e13:
            # is it pointing toward us?
            if beaming_cut(Pf) == 0:
                # print("Did not pass beaming test")
                continue
        
            # is it observable?
            lum, s_den1GHz = Lum(Pf, Pdot, dist_earth, alpha=0.48, muLcorr=0.0, sigLcorr=0.8)

            minF1 = min_flux_test1(Pf, tau_sc.value, threshold=width_threshold)
            if (minF1 == 0):
                # print("Too broad....")
                continue
            minF2 = min_flux_test2(Pf, DM.value, beta=2, tau=1.0e-2, DM0=60, DM1=7.5, G=0.64, Trec=21, d_freq=10e6, tobs=200, SN=10, Tsky=1.0)
            # print("Flux stuff \t", s_den1GHz, minF1, minF2)
            if (s_den1GHz < minF2):
                # print("Flux too low...")
                continue
            
        else:
            # print("magnetar?")
            s_den1GHz = 1.0
    
        #final_list.append([Bf, Pf, Pdot, ThetaF, dist_earth, s_den1GHz, DM.value, tau_sc.value, xfin[0], xfin[1], xfin[2]])
        final_list.append([Pf, Pdot])
        
    return np.asarray(final_list)


def mcmc_func_minimize(real_samples, max_T=1e7):

    def lnprior(theta):
        mu_P, mu_B, sig_P, sig_B, cov_PB = theta
        if 0.0 < sig_P < 10.0 and 0.0 < sig_B < 10.0 and 0.0 < cov_PB < 10.0:
            return 0.0
        return -np.inf
    
    def likelihood_func(theta, Nsamps, real_samples, max_T):
        
        lp = lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        # sample points
        mu_P, mu_B, sig_P, sig_B, cov_PB = theta
        mean = [mu_P, mu_B]
        cov = [[sig_P, cov_PB], [cov_PB, sig_B]]
        x, y = np.random.multivariate_normal(mean, cov, Nsamps).T
        P_in = np.exp(x)
        B_in = np.exp(y)
        data_in = np.column_stack((B_in, P_in))
        
        ages = np.random.randint(0, int(max_T), len(B_in))
        # run forward model
        out_pop = simulate_pop(Nsamps, ages, beta=6e-40, tau_Ohm=10.0e6, width_threshold=0.1, pulsar_data=data_in)
        P_out = out_pop[:, 0]
        Pdot_out = out_pop[:, 1]
        
        Pdotrange = np.logspace(-23, -9, 50)
        Prange = np.linspace(0.01, 20, 50)
        n_sim = len(P_out)
        n_dat = len(real_samples[:,1])
        # print(np.min(Pdot_out), np.max(Pdot_out), np.min(P_out), np.max(P_out))
        cnt = 0
        log_q = 0.0
        for i in range(len(Pdotrange)):
            for j in range(len(Prange)):
                cond1 = Pdot_out < Pdotrange[i]
                cond2 = P_out < Prange[j]
                
                cdf_1 = np.sum(np.all( np.column_stack((cond1, cond2)), axis=1)) / n_sim
                
                cond1_d = real_samples[:,1] < Pdotrange[i]
                cond2_d = real_samples[:,0] < Prange[j]
                cdf_2 = np.sum(np.all( np.column_stack((cond1_d, cond2_d)), axis=1)) / n_dat
                
                log_q += (cdf_1 - cdf_2)**2
        print("log_q", log_q)
        return log_q


    ndim, nwalkers = 5, 100
    # params: mu_P, mu_B, sig_P, sig_B, cov_PB
    central_v = np.array([np.log(0.3), np.log(10**12.95), 0.1, 0.4, 0.0])
    pos = [central_v + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

    Nsamples=5000
    sampler = emcee.EnsembleSampler(nwalkers, ndim, likelihood_func, args=[Nsamples, real_samples, max_T])
    sampler.run_mcmc(pos, 5000, progress=True)
    
    burn_in = 100
    samples = sampler.chain[:, burn_in:, :].reshape((-1, ndim))
    
    
    fig = corner.corner(samples, labels=[r"$\log_{\mu_P}$", r"$\log_{\mu_B}$", r"$\log_{\sigma_p}$", r"$\log_{\sigma_B}$", r"$\log_{\sigma_BP}$"])
    fig.savefig("triangle_TEST.png")

    return
 


def train_nf(true_pop, tau_ohmic=1e7):
    
    base = nf.distributions.DiagGaussian(2)
    # base = nf.distributions.UniformGaussian(2, [1], torch.tensor([1., 20.0]))
    LR = 1e-4
    
    num_layers = 32
    flows = []
    for i in range(num_layers):
        # Neural network with two hidden layers having 64 units each
        # Last layer is initialized by zeros making training more stable
        param_map = nf.nets.MLP([1, 64, 64, 2], init_zeros=False)
        # Add flow layer
        flows.append(nf.flows.AffineCouplingBlock(param_map))
        # Swap dimensions
        flows.append(nf.flows.Permute(2, mode='swap'))
#    K = 30
#    flow_layers = []
#    for i in range(K):
#        flow_layers += [nf.flows.CircularAutoregressiveRationalQuadraticSpline(2, 1, 512, [1], num_bins=10,
#                                                                               tail_bound=torch.tensor([5., np.pi]),
#                                                                               permute_mask=True)]

    model = NormalizingFlow_2(base, flows)

    # Move model on GPU if available
    enable_cuda = True
    device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
    model = model.to(device)
    
    # Train model
    max_iter = 1000
    num_samples = 10000
    num_samples_pop = 1500
    show_iter = 20

    torch.set_grad_enabled(True)
    loss_hist = np.array([])

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter)
    optimizer.zero_grad()
    for it in tqdm(range(max_iter)):
        
        
        # Compute loss
        # x needs to be batch from NS pop.
        indx_ns = np.random.choice(range(1, len(true_pop[:,0])), num_samples_pop, replace=False)
        #loss = model.custom_ns_population(num_samples, true_pop[indx_ns], max_T=tau_ohmic*5, tau_Ohm=tau_ohmic)
        loss = model.custom_ns_population(num_samples, true_pop, max_T=tau_ohmic*5, tau_Ohm=tau_ohmic)
        loss.requires_grad = True
        # Do backprop and optimizer step
        print(loss)
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            optimizer.step()
        
        # Log loss
        loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
        
        # Plot learned model
        if (it + 1) % show_iter == 0:
            z, log_q_ = model.q0(1000)
            
            f, ax = plt.subplots()
            ax.scatter(np.exp(z[:,0].detach().numpy()), np.exp(np.log(10**12.0) + z[:, 1].detach().numpy()), s=1)
            Btrue = true_pop[:,0]
            B_true = 1e12 * np.sqrt((true_pop[:,1] / 1e-15)  * true_pop[:,0])
            ax.scatter(true_pop[:,0], B_true, s=1)
            ax.set_xscale("log")
            ax.set_yscale("log")
            # f, ax = plt.subplots(1, 2, figsize=(15, 7))
            # ax[0].hist(np.exp(z[:,0]), bins=20, histtype='step')
            # ax[0].hist(np.exp(z[:,0]), bins=20, histtype='step')
            # ax[1].hist((np.log(10**12.5) + z[:,1]) / 2.718281, bins=20, histtype='step')
            plt.show()
                # Iterate scheduler
        scheduler.step()


    return


# train_nf(true_pop)
mcmc_func_minimize(true_pop)
