using Random
using Printf
using PyCall
using OrdinaryDiffEq
using Statistics
using SpecialFunctions: erfinv, erf
using DelimitedFiles
using Distributions
using LSODA
using StatsBase
using Suppressor

# pyimport_conda("astropy.units", "")
# Load required Python libraries
u = pyimport("astropy.units")
pygedm = pyimport("pygedm")



run_magnetars = false
fileName = "Test_Run"

Pmin=0.05
Pmax=0.75
Bmin=1e12
Bmax=3e13

sigP_min=0.02
sigP_max=0.4
sigB_min=0.1
sigB_max=1.2

Npts_P=5
Npts_B=5
NPts_Psig=5
NPts_Bsig=5

tau_ohmic = 10.0e6  # yrs
max_T = 5.0 * tau_ohmic

Nsamples = 20000 # samples per point



filePulsars = open(readdlm,"psrcat_tar/store_out/Pulsar_Population.txt")
fileMagnetars = open(readdlm, "psrcat_tar/store_out/Magnetar_Population.txt")

if !run_magnetars
    true_pop = filePulsars[:, 2:3]
    B_max = 4.4e13
    B_min = 1e10
else
    true_pop = vcat(filePulsars[:, 2:3], fileMagnetars[:, 2:3])
    B_max = 1e15
    B_min = 1e10
end

N_pulsars_tot = 3389 # this is ATNF number
print("Number pulsars in sample \t", N_pulsars_tot, "\n")

# cut data
true_pop = true_pop[true_pop[:,2] .> 0, :]
B_true = 1e12 .* sqrt.((true_pop[:, 2] ./ 1e-15) .* true_pop[:, 1])
true_pop = true_pop[(B_true .> B_min) .& (B_true .< B_max), :]

rval = cor(true_pop[:,1], true_pop[:,2])
# print("Correlation Coeff \t ", rval, "\n")

NS_formationrateL = 1.0  # ~ 2 per century [be conservative]
NS_formationrateH = 4.0  # ~ 3 per century [be conservative]
num_pulsarsL = NS_formationrateL / 1e2 * max_T
num_pulsarsH = NS_formationrateH / 1e2 * max_T
println("Estimated number of pulsars formed in the last ", max_T, " years: ", num_pulsarsL, "\t", num_pulsarsH, "\n")


function beaming_cut(P)
    f_per = (9 * log10.(P / 10.0)^2 + 3.0) * 1e-2
    if f_per > 1
        f_per = 1.0
    end
    is_pointing = sample([0, 1], Weights([1.0 - f_per, f_per]))
    return is_pointing
end

function Lum(P, Pdot, dist; alpha=0.48, muLcorr=0.0, sigLcorr=0.8)
    L0 = 5.69e6
    L1Ghz = L0 * 10^randn() * (Pdot / P^3)^alpha
    return L1Ghz, L1Ghz / dist^2
end

function sample_location(age; diskH=0.5, diskR=10.0)
    hh = rand(range(-diskH, diskH, 1000))
    rr = diskR * sqrt(rand())
    phi_loc = 2 * pi * rand()
    x_cart = [rr * cos(phi_loc), rr * sin(phi_loc), hh]
    Vel1d = 0.9 * (2 * sqrt(190.0) * sqrt(log(1 / (1 - rand())))) + 0.1 * (2 * sqrt(786.0) * sqrt(log(1 / (1 - rand()))))
    theta = acos(1.0 - 2.0 * rand())
    phi = 2 * pi * rand()
    vel_vec = Vel1d * [cos(phi) * sin(theta), sin(theta) * sin(phi), cos(theta)]
    xpos = x_cart + vel_vec * age * (3.24078e-17 * 3.15e7)
    return xpos
end

function min_flux_test1(P, tau_sc; threshold=0.1)
    if tau_sc / P > threshold
        return 0
    else
        return 1
    end
end

function min_flux_test2(P, DM; beta=2, tau=1.0e-2, DM0=60, DM1=7.5, G=0.64, Trec=21, d_freq=10e6, tobs=200, SN=10, Tsky=1.0)
    wid = 0.05 * P
    Smin = SN * beta * (Trec + Tsky) / (G * 2 * d_freq * tobs) * 1e3
    w_e = sqrt.(wid^2 + (beta * tau)^2 + (tau * DM / DM0)^2 + (tau * (DM - DM1) / DM1)^2)
    if w_e .>= P
        return 1e100
    else
        return Smin * sqrt.(w_e / (P - w_e))
    end
end

function evolve_pulsar(B0, P0, Theta_in, age; n_times=1e2, beta=6e-40, tau_Ohm=10.0e6)
    times = exp10.(range(0, stop=log10(age), length=Int(n_times)))
    
    y0 = [P0, mod(Theta_in, pi/2)]
    Mvars = [beta, tau_Ohm, B0]
    tspan = (1, age)
    saveat = (tspan[2] .- tspan[1]) ./ 10
    
    prob = ODEProblem(RHS!, y0, tspan, Mvars, reltol=1e-5, abstol=1e-4)
    # sol = @suppress solve(prob, lsoda(), saveat=saveat)
    sol = solve(prob, Vern6(), saveat=saveat)
    
    
    Bf = B0 .* exp.(- age ./ tau_Ohm);
    Pf = sol.u[end][1]
    if Pf .< P0
        Pf = P0
    end
    Theta_f = sol.u[end][2]
    
    Pdot = beta .* Bf.^2 ./ Pf .* (1.0 .+ 1.0 .* sin.(Theta_f).^2)
    return Bf, Pf, Theta_f, Pdot
end


function RHS!(du, u, Mvars, t)
    beta, tau_Ohm, B0 = Mvars
    
    B = B0 .* exp.(- t ./ tau_Ohm);
    
    if B < 1e9
        return [0.0, 0.0, 0.0]
    end
    P = u[1]
    chi = mod(u[2], pi/2)
    
    alive = (B / P^2 > 0.34 * 1e12)
    
    # du[1] = -1.0 / tau_Ohm
    
    du[1] = beta * B.^2 ./ P * (alive * 1.0 + 1.0 * sin(chi)^2) .* 3.1536e+7
    du[2] = -1.0 * beta * B^2 / P^2 * sin(chi) * cos(chi) .* 3.1536e+7
    
    return
end

function draw_Bfield_lognorm()
    mu_B = 12.95
    sig_B = 0.55
    return 10^(mu_B + sqrt(2) * sig_B * erfinv(2 * rand() - 1.0))
end

function draw_period_norm()
    mu_P = 0.3
    sig_P = 0.15
    val = mu_P + sqrt(2) * sig_P * erfinv(2 * rand() - 1.0)
    if val <= 0
        return draw_period_norm()
    else
        return val
    end
end

function draw_chi(Nsamps)
    return acos.(1.0 .- 2.0 .* rand(Nsamps))
end

function simulate_pop(num_pulsars, ages; beta=6e-40, tau_Ohm=10.0e6, width_threshold=0.1, pulsar_data=nothing)
    
    final_list = []
    Theta_in = draw_chi(length(ages))
    
    for i in 1:length(ages)
        
        if typeof(pulsar_data) == Nothing
            B0 = draw_Bfield_lognorm()
            P0 = draw_period_norm()
        else
            B0 = pulsar_data[i, 1]
            P0 = pulsar_data[i, 2]
        end
        
        Bf, Pf, ThetaF, Pdot = evolve_pulsar(B0, P0, Theta_in[i], ages[i], beta=beta, tau_Ohm=tau_Ohm)
        
        Bdeath = 0.34 * 1e12 * Pf.^2
        if Bf <= Bdeath
            # print("death? \n")
            continue
        end
        xfin = sample_location(ages[i], diskH=0.5, diskR=10.0)
        if abs(xfin[3]) > 0.5
            # print("z \t", xfin, " \n")
            continue
        end
        xE = [0.0, 8.3, 0.0]
        xfin_shift = xfin .- xE
        dist_earth = sqrt.(sum(xfin_shift.^2))
        GC_b = asin.(xfin[3] / dist_earth)
        GC_l = atan.(xfin_shift[1], xfin_shift[2])
        DM, tau_sc = pygedm.dist_to_dm(GC_l, GC_b, dist_earth * 1e3 * u.pc, method="ymw16")
        if Bf < 4.4e13
            
            if beaming_cut(Pf) == 0
                continue
            end
            lum, s_den1GHz = Lum(Pf, Pdot, dist_earth, alpha=0.48, muLcorr=0.0, sigLcorr=0.8)
            
            minF1 = min_flux_test1(Pf, tau_sc[1], threshold=width_threshold)
            if minF1 == 0
                continue
            end
            minF2 = min_flux_test2(Pf, DM[1], beta=2, tau=1.0e-2, DM0=60, DM1=7.5, G=0.64, Trec=21, d_freq=10e6, tobs=200, SN=10, Tsky=1.0)
            if s_den1GHz < minF2
                continue
            end
        else
            s_den1GHz = 1.0
        end
        
        push!(final_list, [Pf, Pdot])
    end
    
    return reduce(hcat, final_list)'
end


function likelihood_func(theta, real_samples, rval; npts_cdf=100)
    mu_P, mu_B, sig_P, sig_B, cov_PB = theta
    mean = [mu_P, mu_B]
    cov = [sig_P cov_PB; cov_PB sig_B]
    dist = MvNormal(mean, cov)
    out_samps = rand(dist, Nsamples)'
    
    
    P_in = 10 .^(out_samps[:,1])
    B_in = 10 .^(out_samps[:,2])
    data_in = hcat(B_in, P_in)
    ages = rand(0:max_T, length(B_in))
    
    out_pop = simulate_pop(Nsamples, ages, beta=6e-40, tau_Ohm=10.0e6, width_threshold=0.1, pulsar_data=data_in)
    Obs_pulsarN_L = length(out_pop[:,1]) ./ Nsamples .* num_pulsarsL
    Obs_pulsarN_H = length(out_pop[:,1]) ./ Nsamples .* num_pulsarsH
    
    
    if (N_pulsars_tot .< Obs_pulsarN_L)||(N_pulsars_tot .> Obs_pulsarN_H)
        print("Pred Low, Pred High, Actual \t", Obs_pulsarN_L , "\t", Obs_pulsarN_H, "\t", N_pulsars_tot, "\n" )
        return 1e-100
    end
    
    if isempty(out_pop)
        print("Empty?? \n")
        return -Inf
    else
        P_out = out_pop[:, 1]
        Pdot_out = out_pop[:, 2]
        Pdotrange = exp10.(range(-23, stop=-9, length=npts_cdf))
        
        Prange = exp10.(range(-2.5, stop=log10(20), length=npts_cdf))
        n_sim = length(P_out)
        n_dat = length(real_samples[:, 1])
        cnt = 0
        log_q = []
        for i in 1:length(Pdotrange)
            for j in 1:length(Prange)
                cond1 = Pdot_out .< Pdotrange[i]
                cond2 = P_out .< Prange[j]
                cdf_1 = sum(all(hcat(cond1, cond2), dims=2)) / n_sim
                
                cond1_d = real_samples[:, 2] .< Pdotrange[i]
                cond2_d = real_samples[:, 1] .< Prange[j]
                cdf_2 = sum(all(hcat(cond1_d, cond2_d), dims=2)) / n_dat
                
                push!(log_q, abs(cdf_1 - cdf_2))
            end
        end
        Dval = maximum(log_q)
        
        neff = n_dat .* n_sim ./ (n_dat .+ n_sim)
        lam = sqrt.(neff) .* Dval ./ (1.0 .+ sqrt.(1.0 .- rval.^2) .* (0.25 .- 0.75 ./ sqrt.(neff)))
        
        Qval = 0.0
        for i in 1:10
            Qval += (-1).^(i-1) .* exp.(-2 .* i.^2 .* lam.^2)
        end
        
        # println("log_q: ", log_qM, "\n")
        # println("log_q: ", log_q, "\n")
        # return -log_qM
        print("Dval \t", Dval, "  p(D > Dobs) \t", Qval, "\n")
        return Qval
    end
end



function hard_scan(real_samples, rval; max_T=1e7, Pmin=0.05, Pmax=0.75, Bmin=1e12, Bmax=5e13, sigP_min=0.05, sigP_max=0.4, sigB_min=0.1, sigB_max=1.2, Npts_P=5, Npts_B=5, NPts_Psig=5, NPts_Bsig=5)
    Pmu_scan = range(log10.(Pmin), stop=log10.(Pmax), length=Npts_P)
    Bmu_scan = range(log10.(Bmin), stop=log10.(Bmax), length=Npts_B)
    Psig_scan = range(sigP_min, stop=sigP_max, length=NPts_Psig)
    Bsig_scan = range(sigB_min, stop=sigB_max, length=NPts_Bsig)
    qval_L = []
    for i1 in 1:length(Pmu_scan)
        for i2 in 1:length(Bmu_scan)
            for i3 in 1:length(Psig_scan)
                for i4 in 1:length(Bsig_scan)
                    theta = [Pmu_scan[i1], Bmu_scan[i2], Psig_scan[i3], Bsig_scan[i4], 0.0]
                    print(theta, "\n")
                    qval = likelihood_func(theta, real_samples, rval)
                    push!(qval_L, [Pmu_scan[i1], Bmu_scan[i2], Psig_scan[i3], Bsig_scan[i4], 0.0, qval])
                    
                end
            end
        end
    end
    return qval_L
end

outputTable = hard_scan(true_pop, rval, max_T=max_T, Pmin=Pmin, Pmax=Pmax, Bmin=Bmin, Bmax=Bmax, sigP_min=sigP_min, sigP_max=sigP_max, sigB_min=sigB_min, sigB_max=sigB_max, Npts_P=Npts_P, Npts_B=Npts_B, NPts_Psig=NPts_Psig, NPts_Bsig=NPts_Bsig)

writedlm(fileName*".dat", outputTable)
