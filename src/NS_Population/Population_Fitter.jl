using Random
using Printf
using PyCall
using OrdinaryDiffEq
using Statistics
using SpecialFunctions: erfinv, erf, gamma
using DelimitedFiles
using Distributions
using LSODA
using StatsBase
using Suppressor
using Dates
using Distributed
using SharedArrays
using Optimization
using OptimizationOptimJL
# using Optim
import AffineInvariantMCMC
include("YMW16.jl")
# using Flux
# using SciMLSensitivity

# pyimport_conda("astropy.units", "")
# Load required Python libraries
# u = pyimport("astropy.units")
# pygedm = pyimport("pygedm")


# Random.seed!(1235)

function weibull_mod(P, J, K)
    return J ./ (K .* P.^2) .* (1.0 ./ (K .* P)).^(J .- 1.0) .* exp.(- 1.0 ./ (K .* P).^J)
end

function beaming_cut(P, ThetaF)
    # one way
    # f_per = (9 * log10.(P / 10.0)^2 + 3.0) * 1e-2
    # alternate way
    rhob = sqrt.(9 .* pi * 300.0 ./ (2 .* P) ./ 2.998e5)
    if (ThetaF - rhob) > 0.0
        f_per = sin.(ThetaF) .* sin.(rhob) .* 2
    else
        f_per = 1.0 .- cos.(ThetaF .+ rhob)
    end
    # print(f_test, "\t", f_per, "\n")
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
    rhob = sqrt.(9 .* pi * 300.0 ./ (2 .* P) ./ 2.998e5)
    return L1Ghz, L1Ghz / dist^2 ./ (4 .* pi .* (1 .- cos.(rhob)))
end

function radial_cdf(r)
    # based on  2402.11428
    cdf_v = zeros(length(r))
    for i in 1:length(r)
        cdf_v[i] = 1 .- 1.16247e-6 .* gamma(10.03, 4.36172 .+ 1.16003 .* r[i]) .+ 5.3303e-7 .* gamma(11.03, 4.36172 .+ 1.16003 .* r[i]) .- 6.11031e-8 .* gamma(12.03, 4.36172 .+ 1.16003 .* r[i])
    end
    return cdf_v
end

function sample_radial_coord()
    rlist = range(0.0, 15.0, 400)
    cdf_v = radial_cdf(rlist)
    return rlist[argmin(abs.(cdf_v .- rand()))]
end

function sample_location(age; diskH=0.18, diskR=10.0)
    # hh = rand(range(-diskH, stop=diskH, length=1000))
    hh = diskH .* log.(1.0 ./ (1.0 .- rand()))
    if rand() .> 0.5
        hh *= -1.0
    end
    # rr = diskR * sqrt(rand())
    rr = sample_radial_coord()
    
    phi_loc = 2 * pi * rand()
    x_cart = [rr * cos(phi_loc), rr * sin(phi_loc), hh]
    
    # Vel1d = 0.9 * (2 * sqrt(190.0) * sqrt(log(1 / (1 - rand())))) + 0.1 * (2 * sqrt(786.0) * sqrt(log(1 / (1 - rand()))))
    # theta = acos(1.0 - 2.0 * rand())
    # phi = 2 * pi * rand()
    # vel_vec = Vel1d * [cos(phi) * sin(theta), sin(theta) * sin(phi), cos(theta)]
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

function min_flux_test2(P, DM, chi, t_scat; beta=2, tau=1.0e-2, DM0=60, DM1=7.5, G=0.75, Trec=21, d_freq=288e6, f_ch=3000e3, tobs=2000, SN=9, Tsky=1.0)
    
    rb = sqrt.(9 .* pi * 300.0 ./ (2 .* P) ./ 2.998e5)
    # alph = rand() .* rb .+ chi
    # w_int = 4 .* P .* asin.(sqrt.( abs.( sin.(0.5 .* (rb .+ (alph .- chi))) .* sin.(0.5 .* (rb .- (alph .- chi))) ./ (sin.(chi) .* sin.(alph)))) )
    w_int = 2 .* rb ./ cos.(chi) .* P
    tau_samp = 200e-6
    tau_dm = 0.3.^2 ./ (pi * 5.11e5) .* f_ch ./ (1.4e9.^3) .* DM .* (3.086e18) .* 2.998e10.^2 .* 6.58e-16

    
    w_ob = sqrt.(w_int.^2 .+ tau_samp.^2 .+ tau_dm.^2 .+ t_scat.^2)
    
    # wid = 0.05 * P
    
    if w_ob .>= P
        return 1e100
    else
        Smin = SN * beta * (Trec + Tsky) / (G * sqrt.(2 * d_freq * tobs)) * sqrt.(w_ob / (P - w_ob)) .* (P ./ w_int) .* 1e3
        return Smin
    end
end

function evolve_pulsar(B0, P0, Theta_in, age; n_times=1e2, beta=6e-40, tau_Ohm=10.0e6)
    
    y0 = [P0, mod(Theta_in, pi/2)]
    Mvars = [beta, tau_Ohm, B0]
    tspan = (0, age)
    saveat = (tspan[2] .- tspan[1]) ./ n_times
    
    Bf = B0 .* exp.(- age ./ tau_Ohm);
    
    prob = ODEProblem(RHS!, y0, tspan, Mvars, reltol=1e-3, abstol=1e-6, dtmin=1e-5, force_dtmin=true)
    
    condition_r(u,lnt,integrator) = u[2]
    function affect_r!(integrator)
        integrator.u[2] = 0.0
    end
    
    Pdeath = sqrt.(Bf ./ (0.34 * 1e12))
    condition_dead(u,lnt,integrator) = u[1] .> Pdeath
    function affect_dead!(integrator)
        terminate!(integrator)
    end
    cb_d = ContinuousCallback(condition_dead, affect_dead!)
    cb_r = ContinuousCallback(condition_r, affect_r!)
    cb_all = CallbackSet(cb_r, cb_d)
    
    sol = solve(prob, Vern6(), saveat=saveat, callback=cb_all)
    # sol = solve(prob, Vern6(), saveat=saveat)
    
    
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
    # print(t, "\t", u, "\n")
    beta, tau_Ohm, B0 = Mvars
    
    B = B0 .* exp.(- t ./ tau_Ohm);
    
    # if B < 1e9
    #     return [0.0, 0.0, 0.0]
    # end
    P = u[1]
    chi = mod(u[2], pi/2)
    
    
    alive = (B / P^2 > 0.34 * 1e12)
    
    # du[1] = -1.0 / tau_Ohm
    spin = 1.0
    if u[2] < 0.01
        spin = 0.0
    end
    du[1] = beta * B.^2 ./ P * (alive * 1.0 + 1.0 * sin(chi)^2) .* 3.1536e+7
    du[2] = -spin * beta * B^2 / P^2 * sin(chi) * cos(chi) .* 3.1536e+7

    return
end

function draw_Bfield_lognorm(;muB=12.95, sigB=0.55)
    return 10 .^(muB .+ sqrt(2) .* sigB .* erfinv(2 * rand() .- 1.0))
end

function draw_period_norm(;mu_P=0.3, sig_P=0.15)
    val = mu_P + sqrt(2) * sig_P * erfinv(2 * rand() - 1.0)
    if val <= 0
        return draw_period_norm(mu_P=mu_P, sig_P=sig_P)
    else
        return val
    end
end

function draw_chi(Nsamps)
    return acos.(1.0 .- 2.0 .* rand(Nsamps))
end

function simulate_pop(num_pulsars, ages; beta=6e-40, tau_Ohm=10.0e6, width_threshold=0.1, pulsar_data=nothing, B_min=1e10, B_max=4.4e13)
    
    # final_list = []
    Theta_in = draw_chi(length(ages))
    
    temp_store = zeros(length(ages), 9)
    
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
        # quick check to see if life is easy...
        Bdeath = 0.34 * 1e12 * P0.^2
        if (Bf <= Bdeath)&&(kill_dead)
            continue
        end
        
        Bf, Pf, ThetaF, Pdot = evolve_pulsar(B0, P0, Theta_in[i], ages[i], beta=beta, tau_Ohm=tau_Ohm)
        # robust check
        Bdeath = 0.34 * 1e12 * Pf.^2
        if (Bf <= Bdeath)&&(kill_dead)
            continue
        end
        
        if (Bf < B_min)||(Bf > B_max)
            continue
        end
        
        # GC_b = asin.(xfin[3] / dist_earth)
        # GC_l = atan.(xfin_shift[1], xfin_shift[2])
        
        temp_store[i, :] .= [ages[i], Bf, Pf, Pdot, xfin[1], xfin[2], xfin[3], dist_earth, ThetaF]
    end
    
    temp_store = temp_store[temp_store[:,1] .> 0.0, :]
    
    final_list = zeros(length(temp_store[:,1]), 2)
    test1=0
    test2=0
    for i in 1:length(temp_store[:,1])
        age, Bf, Pf, Pdot, xfin_X, xfin_Y, xfin_Z, dist_earth, ThetaF = temp_store[i, :]
        xfin = [xfin_X xfin_Y xfin_Z]
        
        if beaming_cut(Pf, ThetaF) == 0
            continue
        end
        lum, s_den1GHz = Lum(Pf, Pdot, dist_earth, alpha=0.48, muLcorr=0.0, sigLcorr=0.8)
        
        DM = getDM(xfin .* 1e3, 10000)
        # DM, tau_sc = pygedm.dist_to_dm(GC_l .* 180 ./ pi, GC_b .* 180 ./ pi, dist_earth * 1e3 * u.pc, method="ymw16")
        t_scat_mean = (3.6e-9 .* DM.^2 .* (1 .+ 1.94e-3 .* DM.^2)) .* (1400.0 ./ 327).^(-4.4)
        tau_sc = 10 .^(log10.(t_scat_mean) .+ sqrt(2) .* 0.5 .* erfinv(2 * rand() .- 1.0))
        
        
        minF1 = min_flux_test1(Pf, tau_sc, threshold=width_threshold)
        if minF1 == 0
            test1 += 1
            continue
        end
        minF2 = min_flux_test2(Pf, DM[1], ThetaF, tau_sc)
        
        if s_den1GHz < minF2
            test2 += 1
            continue
        end
        
        
        # push!(final_list, [Pf, Pdot])
        final_list[i, :] = [Pf, Pdot]
    end
    # print(test1, "\t", test2, "\t", length(temp_store[:,1]), "\n")
    final_out = final_list[final_list[:,1] .> 0.0, :]
    return final_out
end


function likelihood_func(theta, real_samples, rval, Nsamples, max_T; npts_cdf=50, tau_Ohm=10.0e6, B_minT=1e10, B_maxT=4.4e13, gauss_approx=true, Pabsmin=1e-3, constrain_birthrate=false, ks_like=true)
    
    if gauss_approx
        mu_P, mu_B, sig_P, sig_B = theta
        mean = [mu_P, mu_B]
        cov = [sig_P.^2 0.0; 0.0 sig_B.^2]
 
        dist = MvNormal(mean, cov)
        out_samps = rand(dist, Nsamples)'
        # P_in = abs.(out_samps[:,1]) #
        P_in = 10 .^ out_samps[:, 1]
        B_in = 10 .^(out_samps[:,2])
        data_in = hcat(B_in, P_in)
    else
        P_shape, mu_B, P_scale, sig_B = theta
        Plist = 10 .^ range(-2.5, 1.5, 1000)
        
        data_in = zeros(Nsamples, 2)
        for i in 1:Nsamples
            # Ptemp = (Pmax .^ (1 .+ Pbeta) .* rand()).^(1.0 ./ (1.0 .+ Pbeta))
            # Pabsmin=1e-3
            # Ptemp = ((-Pmax.^(1 .+ Pbeta) .+ Pabsmin .^(1 .+ Pbeta)) .* (- Pabsmin.^(1 .+ Pbeta) ./ (Pmax.^(1 .+ Pbeta) .- Pabsmin.^(1 .+ Pbeta)) .- rand())).^(1.0 ./ (1.0 .+ Pbeta))
            
            wegt = weibull_mod(Plist, P_shape, P_scale)
            Ptemp = sample(Plist, Weights(wegt), 1)[1]
            Btemp = draw_Bfield_lognorm(muB=mu_B, sigB=sig_B)
            data_in[i, :] = [Btemp Ptemp]
        end
    end
    
    
    ages = rand(1:max_T, length(data_in[:, 1]))
    
    out_pop = simulate_pop(Nsamples, ages, beta=6e-40, tau_Ohm=tau_Ohm, width_threshold=0.1, pulsar_data=data_in, B_min=B_minT, B_max=B_maxT)
    num_out = length(out_pop[:,1])
    num_real = length(real_samples[:, 1])
    pulsar_birth_rate = (Nsamples ./ max_T .* 100) .* (num_real ./ num_out)
    birth_prob = exp.(-(1.63 .- pulsar_birth_rate).^2 ./ (2 .* 0.46.^2))
    # birth_prob = exp.(-(1.63 .- pulsar_birth_rate).^2 ./ (2.0 .* 1.2 .^2))
        
    print("Pulsar Birth Rate (per century) \t", pulsar_birth_rate, "\t", num_out, "\n")
    
    Pbins = 10 .^ range(-2, stop=1.2, length=20)
    Pdotbins = 10 .^ range(-19, stop=-11, length=20)
    Dval = 0.0
    if isempty(out_pop)
        print("Empty?? \n")
        return -Inf, 0.00
    else
        
        P_out = out_pop[:, 1]
        Pdot_out = out_pop[:, 2]
        
        n_sim = length(P_out)
        n_dat = length(real_samples[:, 1])
        cnt = 0
        log_q = []
        neff = num_real .* num_out ./ (num_out .+ num_real)
        
        if !ks_like
            for i in 1:(length(Pbins))
                for j in 1:(length(Pdotbins) )
                    cond1 = Pdot_out .< Pdotbins[j]
                    # cond2 = Pdot_out .>= Pdotbins[j]
                    cond3 = P_out .< Pbins[i]
                    # cond4 = P_out .>= Pbins[i]
                    cdf_sim = sum(all(hcat(cond1, cond3), dims=2)) ./ num_out
                    

                    cond1 = real_samples[:, 2] .< Pdotbins[j]
                    # cond2 = real_samples[:, 2] .>= Pdotbins[j]
                    cond3 = real_samples[:, 1] .< Pbins[i]
                    # cond4 = real_samples[:, 1] .>= Pbins[i]
                    cdf_real = sum(all(hcat(cond1, cond3), dims=2)) ./ num_real
                    
                    Dval += sqrt.(neff) .* abs.(cdf_sim .- cdf_real)
                    
                end
            end
            if Dval > 100
                Dval /= 10.0
                Dval += 100.0
            end
            
            Qval = exp.(- Dval)
        else
            for i in 1:length(P_out)
                
                # Q1
                cond1 = Pdot_out .< Pdot_out[i]
                cond2 = P_out .< P_out[i]
                cdf_1 = sum(all(hcat(cond1, cond2), dims=2)) / num_out
                
                cond1_d = real_samples[:, 2] .< Pdot_out[i]
                cond2_d = real_samples[:, 1] .< P_out[i]
                cdf_2 = sum(all(hcat(cond1_d, cond2_d), dims=2)) / num_real
                push!(log_q, abs(cdf_1 - cdf_2))
                
                # Q2
                cond1 = Pdot_out .< Pdot_out[i]
                cond2 = P_out .> P_out[i]
                cdf_1 = sum(all(hcat(cond1, cond2), dims=2)) / num_out
                
                cond1_d = real_samples[:, 2] .< Pdot_out[i]
                cond2_d = real_samples[:, 1] .> P_out[i]
                cdf_2 = sum(all(hcat(cond1_d, cond2_d), dims=2)) / num_real
                push!(log_q, abs(cdf_1 - cdf_2))
                
                # Q3
                cond1 = Pdot_out .> Pdot_out[i]
                cond2 = P_out .< P_out[i]
                cdf_1 = sum(all(hcat(cond1, cond2), dims=2)) / num_out
                
                cond1_d = real_samples[:, 2] .> Pdot_out[i]
                cond2_d = real_samples[:, 1] .< P_out[i]
                cdf_2 = sum(all(hcat(cond1_d, cond2_d), dims=2)) / num_real
                push!(log_q, abs(cdf_1 - cdf_2))
                
                # Q4
                cond1 = Pdot_out .> Pdot_out[i]
                cond2 = P_out .> P_out[i]
                cdf_1 = sum(all(hcat(cond1, cond2), dims=2)) / num_out
                
                cond1_d = real_samples[:, 2] .> Pdot_out[i]
                cond2_d = real_samples[:, 1] .> P_out[i]
                cdf_2 = sum(all(hcat(cond1_d, cond2_d), dims=2)) / num_real
                push!(log_q, abs(cdf_1 - cdf_2))
                
            end
            Dval = maximum(log_q)
            Dval = sum(log_q)
            
            neff = n_dat .* n_sim ./ (n_dat .+ n_sim)
            lam = sqrt.(neff) .* Dval ./ (1.0 .+ sqrt.(1.0 .- rval.^2) .* (0.25 .- 0.75 ./ sqrt.(neff)))
            
            Qval = 0.0
            for i in 1:10
                Qval += 2.0 .* (-1).^(i-1) .* exp.(-2 .* i.^2 .* lam)
            end
        end
        
        
        
        if (pulsar_birth_rate > 4.0)&&constrain_birthrate
            Qval *= birth_prob
        end
        print("Dval \t", Dval, "  p(D > Dobs) \t", Qval, "\n")
        return Dval, Qval
    end
end



function minimization_scan(real_samples, rval; max_T=1e7, Nsamples=100000, Phigh=0.0, Plow=-2.0, LBhigh=log10.(3e13), LBlow=log10.(1e12), sPlow=0.05, sPhigh=1.0, sBlow=0.1, sBhigh=1.2, numwalkers=5, Nruns=1000, tau_Ohm=10.0e6, B_minT=1e10, B_maxT=4.4e13, gauss_approx=true, Pabsmin=1e-3, constrain_birthrate=false)
    
    maxV = 1e20
    maxParams = nothing
    
    function prior(theta)
        
        if gauss_approx
            Pv, Bv, sP, sB = theta
            if (Plow .< Pv .< Phigh)&&(LBlow .< Bv .< LBhigh)&&(sPlow .< sP .< sPhigh)&&(sBlow .< sB .< sBhigh)
                return 0.0
            end
        else
            P_shape, Bv, P_scale, sB = theta
            if (0.0 .< P_shape .< 10.0)&&(LBlow .< Bv .< LBhigh)&&(0.0 .< P_scale .< 50.0)&&(sBlow .< sB .< sBhigh)
                return 0.0
            end
        end
        return -Inf
    end
    
    function loss(xIn)
        
        Dval, qval = likelihood_func(xIn, real_samples, rval, Nsamples, max_T, tau_Ohm=tau_Ohm, B_minT=B_minT, B_maxT=B_maxT, gauss_approx=gauss_approx, Pabsmin=Pabsmin, constrain_birthrate=constrain_birthrate)
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
    x0 = rand(4, numwalkers)
    
    if gauss_approx
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
        

    else
        
        len_B = abs.(LBhigh .- LBlow)
        len_sB = abs.(sBlow .- sBhigh)
        
        x0[2, :] .*= len_B
        x0[2, :] .+= LBlow
        x0[4, :] .*= len_sB
        x0[4, :] .+= sBlow
        
        len_Pshape = 10.0
        len_Pscale = 50.0
        x0[1, :] .*= len_Pshape
        x0[3, :] .*= len_Pscale
        

    end
    
    chain, llhoodvals = AffineInvariantMCMC.sample(log_probability, numwalkers, x0, 100, 1)
    
    chain, llhoodvals = AffineInvariantMCMC.sample(log_probability, numwalkers, chain[:, :, end], Nruns, 1)
    flatchain, flatllhoodvals = AffineInvariantMCMC.flattenmcmcarray(chain, llhoodvals)
    aM = argmax(llhoodvals)
    return [llhoodvals[aM], chain[:, aM], transpose(flatchain), flatllhoodvals]
end


function main(run_analysis, run_plot_data, tau_ohmic; Nsamples=10000000, max_T_f=5.0, fileName="Test_Run", xIn=[0.05, log10.(1.4e13), 0.05, 0.65], run_magnetars=false, kill_dead=false,  Pmin=0.05, Pmax=0.75, Bmin=1e12, Bmax=5e13, sigP_min=0.05, sigP_max=0.4, sigB_min=0.1, sigB_max=1.2, Npts_P=5, Npts_B=5, NPts_Psig=5, NPts_Bsig=5, temp=true, minimizeIt=false, numwalkers=5, Nruns=1, gauss_approx=true, Pabsmin=1e-3, constrain_birthrate=false, maxiters=100, ks_like=true)

    print("Tau \t", tau_ohmic, "\n")
    max_T = max_T_f * tau_ohmic


    
    filePulsars = open(readdlm,"psrcat_tar/store_out/Pulsar_Population.txt")

    if !run_magnetars
        true_pop = filePulsars[:, 2:3]
        B_maxT = 4.4e13
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
    N_pulsars_tot = length(true_pop[:, 1])  # this is ATNF number 3389, post cut ~2131
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
            OUTALL = minimization_scan(true_pop, rval; max_T=max_T, Nsamples=Nsamples, numwalkers=numwalkers, Nruns=Nruns, tau_Ohm=tau_ohmic, B_minT=B_minT, B_maxT=B_maxT, gauss_approx=gauss_approx, constrain_birthrate=constrain_birthrate)
            out_bf = OUTALL[2]
            push!(out_bf, OUTALL[1])
            full_chain = OUTALL[3]
            likeVals = OUTALL[4]
            
            writedlm("output_fits/Best_Fit_"*fileName*".dat", out_bf)
            writedlm("output_fits/MCMC_"*fileName*".dat", full_chain)
            writedlm("output_fits/LLIKE_"*fileName*".dat", likeVals)
        else
            
            
            function wrapL(u0, p)
                print(u0, "\n")
                if gauss_approx
                    if u0[3] < 0.1
                        u0[3] = 0.1
                    end
                else
                    if u0[1] < 0.1
                        u0[1] = 0.1
                    end
                    if u0[1] > 10.0
                        u0[1] = 10.0
                    end
                    if u0[3] < 0.1
                        u0[3] = 0.1
                    end
                    if u0[3] > 50.0
                        u0[3] = 50.0
                    end
                
                end
                
                if u0[2] < 12.0
                    u0[2] = 12
                end
                if u0[2] > 13.5
                    u0[2] = 13.5
                end
                if u0[4] < 0.1
                    u0[4] = 0.1
                end
                
                dV, qV = likelihood_func(u0, true_pop, rval, Nsamples, max_T; npts_cdf=50, tau_Ohm=tau_ohmic, B_minT=B_minT, B_maxT=B_maxT, gauss_approx=gauss_approx, Pabsmin=Pabsmin, constrain_birthrate=constrain_birthrate, ks_like=ks_like)
                print("min val \t ", dV, "\n")
                return dV
            end
            
            if gauss_approx
                u0 = [-0.5 12.7 0.5 0.5]
            else
                u0 = [1.0 12.7 14.0 0.5]
            end
            p = [1.0]

            prob = OptimizationProblem(wrapL, u0, p)
            sol = solve(prob, NelderMead(), maxiters=maxiters)
            print("SOLUTION~~~~~~~~ \n\n", sol, "\n\n")
            writedlm("output_fits/ProperMin_Fit_"*fileName*".dat", sol.u)
            outval = wrapL(sol.u, p)
            writedlm("output_fits/ProperMin_Val_"*fileName*".dat", outval)
            
        end
        
    end

    if run_plot_data
        if gauss_approx
            mu_P, mu_B, sig_P, sig_B = xIn
            mean = [mu_P, mu_B]
            cov = [sig_P 0.0; 0.0 sig_B]
            
            
            dist = MvNormal(mean, cov)
            out_samps = rand(dist, Nsamples)'

            
            P_in = abs.(out_samps[:,1]) #
            B_in = 10 .^(out_samps[:,2])
            data_in = hcat(B_in, P_in)
        else
            # Pbeta, mu_B, Pmax, sig_B = xIn
            P_shape, mu_B, P_scale, sig_B = xIn
            data_in = zeros(Nsamples, 2)
            
            Plist = 10 .^ range(-2.5, 1.5, 1000)
        
            for i in 1:Nsamples
         
                wegt = weibull_mod(Plist, P_shape, P_scale)
                Ptemp = sample(Plist, Weights(wegt), 1)[1]
                Btemp = draw_Bfield_lognorm(muB=mu_B, sigB=sig_B)
                # print(Ptemp, "\t", Btemp, "\n")
                data_in[i, :] = [Btemp Ptemp]
            end
            
        end
        
        

        
        ages = rand(1:max_T, length(data_in[:, 1]))
        
        out_pop = simulate_pop(Nsamples, ages, tau_Ohm=tau_ohmic, pulsar_data=data_in)
        # print(out_pop)
        writedlm(fileName*".dat", out_pop)
        writedlm(fileName*"_IN.dat", data_in)
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
