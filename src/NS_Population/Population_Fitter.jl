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
using Dates
using Distributed
using SharedArrays
# using Optim
import AffineInvariantMCMC

# using Flux
# using SciMLSensitivity

# pyimport_conda("astropy.units", "")
# Load required Python libraries
u = pyimport("astropy.units")
pygedm = pyimport("pygedm")


Random.seed!(1235)


function beaming_cut(P)
    f_per = (9 * log10.(P / 10.0)^2 + 3.0) * 1e-2
    if f_per > 1
        f_per = 1.0
    end
    is_pointing = sample([0, 1], Weights([1.0 - f_per, f_per]))
    return is_pointing
end

function Lum(P, Pdot, dist; alpha=0.48, muLcorr=0.0, sigLcorr=0.8)
    L0 = 5.69e6 # mJy / kpc^2
    d = Normal(0.0, 0.8)
    L1Ghz = L0 * 10 .^rand(d,1)[1] * (Pdot / P^3)^alpha
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
    # xpos = x_cart + vel_vec * age * (3.24078e-17 * 3.15e7)
    xpos = x_cart
    return xpos
end

function min_flux_test1(P, tau_sc; threshold=0.1)
    if tau_sc / P > threshold
        return 0
    else
        return 1
    end
end

function min_flux_test2(P, DM; beta=2, tau=1.0e-2, DM0=60, DM1=7.5, G=0.64, Trec=21, d_freq=100e6, tobs=200, SN=10, Tsky=1.0)

    wid = 0.05 * P
    Smin = SN * beta * (Trec + Tsky) / (G * sqrt.(2 * d_freq * tobs)) .* 1e3
    
    w_e = sqrt.(wid^2 + (beta * tau)^2 + (tau * DM / DM0)^2 + (tau * (DM - DM1) / DM1)^2)
    if w_e .>= P
        return 1e100
    else
        return Smin * sqrt.(w_e / (P - w_e))
    end
end

function evolve_pulsar(B0, P0, Theta_in, age; n_times=1e2, beta=6e-40, tau_Ohm=10.0e6)
    
    y0 = [P0, mod(Theta_in, pi/2)]
    Mvars = [beta, tau_Ohm, B0]
    tspan = (0, age)
    saveat = (tspan[2] .- tspan[1]) ./ n_times
    
    Bf = B0 .* exp.(- age ./ tau_Ohm);
    
    prob = ODEProblem(RHS!, y0, tspan, Mvars, reltol=1e-4, abstol=1e-4, dtmin=1e-3, force_dtmin=true)
    
    condition_r(u,lnt,integrator) = u[2]
    function affect_r!(integrator)
        integrator.u[2] = 0.0
    end
    cb_r = ContinuousCallback(condition_r, affect_r!)
    sol = solve(prob, Vern6(), saveat=saveat)
    
    
    Pf = sol.u[end][1]
    if Pf .< P0
        Pf = P0
    end
    Theta_f = sol.u[end][2]
    # print("Sol \t ", Pf, "\t", Theta_f, "\n")
    
    Pdot = beta .* Bf.^2 ./ Pf .* (1.0 .+ 1.0 .* sin.(Theta_f).^2)
    return Bf, Pf, Theta_f, Pdot
end


function RHS!(du, u, Mvars, t)
    
    beta, tau_Ohm, B0 = Mvars
    
    B = B0 .* exp.(- t ./ tau_Ohm);
    
    # if B < 1e9
    #     return [0.0, 0.0, 0.0]
    # end
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
    
    # final_list = []
    Theta_in = draw_chi(length(ages))
    
    temp_store = zeros(length(ages), 7)
    
    Threads.@threads for i in 1:length(ages)
        if typeof(pulsar_data) == Nothing
            B0 = draw_Bfield_lognorm()
            P0 = draw_period_norm()
        else
            B0 = pulsar_data[i, 1]
            P0 = pulsar_data[i, 2]
        end
        
        xfin = sample_location(ages[i], diskH=0.5, diskR=10.0)
        xE = [0.0, 8.3, 0.0]
        xfin_shift = xfin .- xE
        dist_earth = sqrt.(sum(xfin_shift.^2))
        
        Bf = B0 .* exp.(- ages[i] ./ tau_Ohm)
         Bdeath = 0.34 * 1e12 * P0.^2
         if (Bf <= Bdeath)&&(kill_dead)
             continue
         end
        
        Bf, Pf, ThetaF, Pdot = evolve_pulsar(B0, P0, Theta_in[i], ages[i], beta=beta, tau_Ohm=tau_Ohm)
        Bdeath = 0.34 * 1e12 * Pf.^2
        if (Bf <= Bdeath)&&(kill_dead)
            continue
        end
        
        GC_b = asin.(xfin[3] / dist_earth)
        GC_l = atan.(xfin_shift[1], xfin_shift[2])
        
        temp_store[i, :] .= [ages[i], Bf, Pf, Pdot, GC_b, GC_l, dist_earth]
    end
    
    temp_store = temp_store[temp_store[:,1] .> 0.0, :]
    
    final_list = zeros(length(temp_store[:,1]), 2)
    for i in 1:length(temp_store[:,1])
        age, Bf, Pf, Pdot, GC_b, GC_l, dist_earth = temp_store[i, :]

        if beaming_cut(Pf) == 0
            continue
        end
        lum, s_den1GHz = Lum(Pf, Pdot, dist_earth, alpha=0.48, muLcorr=0.0, sigLcorr=0.8)
        
        DM, tau_sc = pygedm.dist_to_dm(GC_l, GC_b, dist_earth * 1e3 * u.pc, method="ymw16")
        
        minF1 = min_flux_test1(Pf, tau_sc[1], threshold=width_threshold)
        if minF1 == 0
            continue
        end
        minF2 = min_flux_test2(Pf, DM[1], beta=2, tau=1.0e-2, DM0=60, DM1=7.5, G=0.64, Trec=21, d_freq=300e6, tobs=1000, SN=5, Tsky=1.0)
        
        if s_den1GHz < minF2
            continue
        end
        
        
        # push!(final_list, [Pf, Pdot])
        final_list[i, :] = [Pf, Pdot]
    end
    
    final_out = final_list[final_list[:,1] .> 0.0, :]
    return final_out
end


function likelihood_func(theta, real_samples, rval, Nsamples, max_T; npts_cdf=50)
    mu_P, mu_B, sig_P, sig_B, cov_PB = theta
    mean = [mu_P, mu_B]
    # cov = [sig_P.^2 cov_PB.^2; cov_PB.^2 sig_B.^2]
    cov = [sig_P.^2 0.0; 0.0 sig_B.^2]
    # print("here 1\n")
    # print("here 2\n\n")
    dist = MvNormal(mean, cov)
    out_samps = rand(dist, Nsamples)'
    
    
    P_in = abs.(out_samps[:,1]) # 10 .^(out_samps[:,1])
    B_in = 10 .^(out_samps[:,2])
    data_in = hcat(B_in, P_in)
    ages = rand(1:max_T, length(B_in))
    
    out_pop = simulate_pop(Nsamples, ages, beta=6e-40, tau_Ohm=10.0e6, width_threshold=0.1, pulsar_data=data_in)
    num_out = length(out_pop[:,1])
#    Obs_pulsarN_L = (num_out - 2*sqrt.(num_out)) ./ Nsamples .* num_pulsarsL
#    Obs_pulsarN_H = (num_out + 2*sqrt.(num_out)) ./ Nsamples .* num_pulsarsH
    # print(num_out, "\n")
    
#    if (N_pulsars_tot .< Obs_pulsarN_L)||(N_pulsars_tot .> Obs_pulsarN_H)
#        print("Pred Low, Pred High, Actual \t", Obs_pulsarN_L , "\t", Obs_pulsarN_H, "\t", N_pulsars_tot, "\n" )
#        return 100, 1e-100
#    end
    
    if isempty(out_pop)
        print("Empty?? \n")
        return -Inf
    else
        P_out = out_pop[:, 1]
        Pdot_out = out_pop[:, 2]
        
        n_sim = length(P_out)
        n_dat = length(real_samples[:, 1])
        cnt = 0
        log_q = []
        for i in 1:length(P_out)
            
            # Q1
            cond1 = Pdot_out .< Pdot_out[i]
            cond2 = P_out .< P_out[i]
            cdf_1 = sum(all(hcat(cond1, cond2), dims=2)) / n_sim
            
            cond1_d = real_samples[:, 2] .< Pdot_out[i]
            cond2_d = real_samples[:, 1] .< P_out[i]
            cdf_2 = sum(all(hcat(cond1_d, cond2_d), dims=2)) / n_dat
            push!(log_q, abs(cdf_1 - cdf_2))
            
            # Q2
            cond1 = Pdot_out .< Pdot_out[i]
            cond2 = P_out .> P_out[i]
            cdf_1 = sum(all(hcat(cond1, cond2), dims=2)) / n_sim
            
            cond1_d = real_samples[:, 2] .< Pdot_out[i]
            cond2_d = real_samples[:, 1] .> P_out[i]
            cdf_2 = sum(all(hcat(cond1_d, cond2_d), dims=2)) / n_dat
            push!(log_q, abs(cdf_1 - cdf_2))
            
            # Q3
            cond1 = Pdot_out .> Pdot_out[i]
            cond2 = P_out .< P_out[i]
            cdf_1 = sum(all(hcat(cond1, cond2), dims=2)) / n_sim
            
            cond1_d = real_samples[:, 2] .> Pdot_out[i]
            cond2_d = real_samples[:, 1] .< P_out[i]
            cdf_2 = sum(all(hcat(cond1_d, cond2_d), dims=2)) / n_dat
            push!(log_q, abs(cdf_1 - cdf_2))
            
            # Q4
            cond1 = Pdot_out .> Pdot_out[i]
            cond2 = P_out .> P_out[i]
            cdf_1 = sum(all(hcat(cond1, cond2), dims=2)) / n_sim
            
            cond1_d = real_samples[:, 2] .> Pdot_out[i]
            cond2_d = real_samples[:, 1] .> P_out[i]
            cdf_2 = sum(all(hcat(cond1_d, cond2_d), dims=2)) / n_dat
            push!(log_q, abs(cdf_1 - cdf_2))
            
        end
        Dval = maximum(log_q)
        
        neff = n_dat .* n_sim ./ (n_dat .+ n_sim)
        lam = sqrt.(neff) .* Dval ./ (1.0 .+ sqrt.(1.0 .- rval.^2) .* (0.25 .- 0.75 ./ sqrt.(neff)))
        
        Qval = 0.0
        for i in 1:10
            Qval += 2.0 .* (-1).^(i-1) .* exp.(-2 .* i.^2 .* lam)
        end
        
        print("Dval \t", Dval, "  p(D > Dobs) \t", Qval, "\n")
        return Dval, Qval
    end
end



function hard_scan(real_samples, rval, Nsamples; max_T=1e7, Pmin=0.05, Pmax=0.75, Bmin=1e12, Bmax=5e13, sigP_min=0.05, sigP_max=0.4, sigB_min=0.1, sigB_max=1.2, Npts_P=5, Npts_B=5, NPts_Psig=5, NPts_Bsig=5)
    
    Pmu_scan = range(Pmin, stop=Pmax, length=Npts_P)
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
                    Dval, qval = likelihood_func(theta, real_samples, rval, Nsamples, max_T)
                  
                    push!(qval_L, [Pmu_scan[i1], Bmu_scan[i2], Psig_scan[i3], Bsig_scan[i4], 0.0, Dval, qval])
                    
                end
            end
        end
    end
    return qval_L
end

function minimization_scan(real_samples, rval; max_T=1e7, Nsamples=100000, Phigh=0.6, Plow=0.02, LBhigh=log10.(3e13), LBlow=log10.(2e12), sPlow=0.05, sPhigh=0.7, sBlow=0.1, sBhigh=1.2, numwalkers=5, Nruns=1000)
    
    maxV = 1e20
    maxParams = nothing
    
    function prior(theta)
        Pv, Bv, sP, sB, cov = theta
        # Pv, Bv, sP, sB = theta
        
        if (Plow .< Pv .< Phigh)&&(LBlow .< Bv .< LBhigh)&&(sPlow .< sP .< sPhigh)&&(sBlow .< sB .< sBhigh)&&(0.0 .< cov .< 0.5)
            return 0.0
        end
        return -Inf
    end
    
    function loss(xIn)
        Dval, qval = likelihood_func(xIn, real_samples, rval, Nsamples, max_T)
        # return Dval
        print(log.(qval), "\t", xIn, "\n")
        return log.(qval)
    end
    
    function log_probability(theta)
        lp = prior(theta)
        if !isfinite.(lp)
            return -Inf
        end
        out = lp .+ loss(theta)
        
        return out
    end
    
    
    # numwalkers=5
    x0 = rand(5, numwalkers)
    
    len_P = abs.(Phigh - Plow )
    len_B = abs.(LBhigh .- LBlow)
    len_sP = abs.(sPlow .- sPhigh)
    len_sB = abs.(sBlow .- sBhigh)
    
    x0[1, :] .*= len_P
    x0[1, :] .+= Plow
    
    x0[2, :] .*= len_B
    x0[2, :] .+= LBlow
    
    x0[3, :] .*= len_sP
    x0[3, :] .+= sPlow
    
    x0[4, :] .*= len_sB
    x0[4, :] .+= sBlow
    
    x0[5, :] .*= 0.5
    
    # Nruns=10
    # print(x0, "\t", numwalkers, "\t", Nruns, "\n")
    chain, llhoodvals = AffineInvariantMCMC.sample(log_probability, numwalkers, x0, Nruns, 1)
    flatchain, flatllhoodvals = AffineInvariantMCMC.flattenmcmcarray(chain, llhoodvals)
    
    aM = argmax(flatllhoodvals)
    print(flatllhoodvals[aM], "\t", flatchain[:, aM], "\n")
    # return maxV, maxParams, chain
    return flatllhoodvals[aM], flatchain[:, aM], flatchain
end


function main(run_analysis, run_plot_data, tau_ohmic; Nsamples=10000000, max_T_f=5.0, fileName="Test_Run", xIn=[0.05, log10.(1.4e13), 0.05, 0.65, 0.0], run_magnetars=false, kill_dead=false,  Pmin=0.05, Pmax=0.75, Bmin=1e12, Bmax=5e13, sigP_min=0.05, sigP_max=0.4, sigB_min=0.1, sigB_max=1.2, Npts_P=5, Npts_B=5, NPts_Psig=5, NPts_Bsig=5, temp=true, minimizeIt=false, numwalkers=5, Nruns=1)

    print("Tau \t", tau_ohmic, "\n")
    max_T = max_T_f * tau_ohmic


    
    filePulsars = open(readdlm,"psrcat_tar/store_out/Pulsar_Population.txt")

    if !run_magnetars
        true_pop = filePulsars[:, 2:3]
        B_maxT = 1e15
        B_minT = 1e10
    else
        fileMagnetars = open(readdlm, "psrcat_tar/store_out/Magnetar_Population.txt")
        true_pop = vcat(filePulsars[:, 2:3], fileMagnetars[:, 2:3])
        B_maxT = 1e15
        B_minT = 1e10
    end



    # cut data
    true_pop = true_pop[true_pop[:,2] .> 0, :]
    B_true = 1e12 .* sqrt.((true_pop[:, 2] ./ 1e-15) .* true_pop[:, 1])
    true_pop = true_pop[(B_true .> B_minT) .& (B_true .< B_maxT), :]
    N_pulsars_tot = length(true_pop[:, 1])  # this is ATNF number 3389
    print("Number pulsars in sample \t", N_pulsars_tot, "\n")


    rval = cor(true_pop[:,1], true_pop[:,2])
    # print("Correlation Coeff \t ", rval, "\n")

    NS_formationrateL = 1.0  # ~ 2 per century [be conservative]
    NS_formationrateH = 4.0  # ~ 3 per century [be conservative]
    num_pulsarsL = NS_formationrateL / 1e2 * max_T
    num_pulsarsH = NS_formationrateH / 1e2 * max_T
    println("Estimated number of pulsars formed in the last ", max_T, " years: ", num_pulsarsL, "\t", num_pulsarsH, "\n")

    # time0=Dates.now()

    if run_analysis
        if minimizeIt
            minP, minV = minimization_scan(true_pop, rval; max_T=max_T, Nsamples=Nsamples, numwalkers=numwalkers, Nruns=Nruns)
            writedlm("output_fits/Best_Fit_"*fileName*".dat", minV)
            writedlm("output_fits/MCMC_"*fileName*".dat", minV)
        else
            outputTable = hard_scan(true_pop, rval, Nsamples, max_T=max_T, Pmin=Pmin, Pmax=Pmax, Bmin=Bmin, Bmax=Bmax, sigP_min=sigP_min, sigP_max=sigP_max, sigB_min=sigB_min, sigB_max=sigB_max, Npts_P=Npts_P, Npts_B=Npts_B, NPts_Psig=NPts_Psig, NPts_Bsig=NPts_Bsig)
            if temp
                writedlm("temp/"*fileName*".dat", outputTable)
            else
                writedlm(fileName*".dat", outputTable)
            end
        end
        
    end

    if run_plot_data
        mu_P, mu_B, sig_P, sig_B, cov_PB = xIn
        mean = [mu_P, mu_B]
        cov = [sig_P cov_PB; cov_PB sig_B]
        # cov = [sig_P 0.0; cov_PB 0.0]
        
        dist = MvNormal(mean, cov)
        out_samps = rand(dist, Nsamples)'

        
        P_in = abs.(out_samps[:,1]) # 10 .^(out_samps[:,1])
        B_in = 10 .^(out_samps[:,2])
        data_in = hcat(B_in, P_in)
        ages = rand(0:max_T, length(B_in))
        
        out_pop = simulate_pop(Nsamples, ages, tau_Ohm=tau_ohmic, pulsar_data=data_in)
        
        writedlm(fileName*".dat", out_pop)
        writedlm(fileName*"_IN.dat", out_samps)
    end

    # time1=Dates.now()
    # print("\n\n Run time: ", time1-time0, "\n")
end

function File_Name_Out(run_analysis, run_plot_data, tau_ohmic; max_T_f=5.0, fileName="Test_Run", xIn=[0.05, log10.(1.4e13), 0.05, 0.65, 0.0], run_magnetars=false, kill_dead=false,  Pmin=0.05, Pmax=0.75, Bmin=1e12, Bmax=5e13, sigP_min=0.05, sigP_max=0.4, sigB_min=0.1, sigB_max=1.2, Npts_P=5, Npts_B=5, NPts_Psig=5, NPts_Bsig=5)

    fileN = "PopFit_TauO_"*string(round(tau_ohmic,sigdigits=2))
    fileN *= "_maxTf_"*string(round(max_T_f,sigdigits=1))
    if run_magnetars
        fileN *= "_MagsYes_"
    end
    if kill_dead
        fileN *= "_KillDeadYes_"
    end
    fileN *= "_Pval_"*string(round(Pmin,sigdigits=1))*"_"*string(round(Pmax,sigdigits=1))
    fileN *= "_Bval_"*string(round(Bmin,sigdigits=1))*"_"*string(round(Bmax,sigdigits=1))
    fileN *= "_sPval_"*string(round(sigP_min,sigdigits=1))*"_"*string(round(sigP_max,sigdigits=1))
    fileN *= "_sBval_"*string(round(sigB_min,sigdigits=1))*"_"*string(round(sigB_max,sigdigits=1))
    fileN *= "_Npts_P_B_sP_sB_"*string(Npts_P)*"_"*string(Npts_B)*"_"*string(NPts_Psig)*"_"*string(NPts_Bsig)*"_"
    
    fileN *= ".dat"
    return fileN
end
