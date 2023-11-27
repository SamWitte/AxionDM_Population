__precompile__()


using ForwardDiff: jacobian
using HDF5
include("Constants.jl")


module RayTracerGR
include("Constants.jl")
include("environment.jl")


using ForwardDiff: gradient, derivative, Dual, Partials, hessian
using OrdinaryDiffEq
using Statistics
using NLsolve
using Random
using LinearAlgebra: cross, det, inv
using SpecialFunctions: erfinv


### Parallelized crossing calculations

struct Crossings
    i1
    i2
    weight
end

"""
Calculate values of matrix X at crossing points
"""
function apply(c::Crossings, A)
    A[c.i1] .* c.weight .+ A[c.i2] .* (1 .- c.weight)
end

"""
calcuates crossings along 2 axis
"""
function get_crossings(A; keep_all=true)
    # Matrix with 2 for upward and -2 for downward crossings
    sign_A = sign.(A)
    #cross = sign_A[:, 2:end] - sign_A[:, 1:end-1]
    cross = sign_A[2:end] - sign_A[1:end-1]
    
    # Index just before crossing
    if keep_all
        i1 = Array(findall(x -> x .!= 0., cross))
    else
        i1 = Array(findall(x -> x .> 0., cross))
    end

    # Index just behind crossing
    #i2 = map(x -> x + CartesianIndex(0, 1), i1)
    i2 = i1 .+ 1

    # Estimate weight for linear interpolation
    weight = A[i2] ./ (A[i2] .- A[i1])

    return Crossings(i1, i2, weight)
end

function magneto_define(Mass_a, plasma_mod, mag_mod, lambdaM, psi; θmQ=0.0, phiQ=0.0, B_DQ=0.0, Mass_NS=1.0, reflect_LFL=false, null_fill=0.0, dead=false, dead_rmax=30.0)
    global func_BField = define_B_model(mag_mod, θmQ=θmQ, phiQ=phiQ, B_DQ=B_DQ, Mass_NS=Mass_NS)
    rho_twist = false
    if mag_mod == "Magnetar"
        rho_twist = true
    end
    global func_Plasma = define_plasma_model(plasma_mod, func_BField, lambdaM=lambdaM, psi=psi, Mass_NS=Mass_NS, rho_twist=rho_twist, reflect_LFL=reflect_LFL, null_fill=null_fill, Mass_a=Mass_a, dead=dead, dead_rmax=dead_rmax)
end

### Parallelized, GPU-backed auto-differentiation ray-tracer

# compute photon trajectories
function func!(du, u, Mvars, lnt)
    @inbounds begin
        t = exp.(lnt);
        
        θm, ωPul, B0, rNS, gammaF, time0, Mass_NS, Mass_a, erg_loc, erg_inf, flat, isotropic, melrose, k0_init = Mvars;
        u[u[:,1] .<= rNS, 1] .= rNS
        if flat
            g_tt, g_rr, g_thth, g_pp = g_schwartz(view(u, :, 1:3), 0.0);
        else
            g_tt, g_rr, g_thth, g_pp = g_schwartz(view(u, :, 1:3), Mass_NS);
        end
        time = time0 .+  t;
        
        # g_tt, g_rr, g_thth, g_pp = g_schwartz(view(u, :, 1:3), Mass_NS);
        k_loc = view(u, :, 4:6) .* erg_inf
        xVals = view(u, :, 1:3)
        erg_loc = erg_inf ./ sqrt.(g_rr)
        
        du[:, 4:6] .= -grad(hamiltonian(seed(xVals), k_loc, time[1], -view(u, :, 7) .* sqrt.(g_rr), θm, ωPul, B0, rNS, Mass_NS, iso=isotropic, melrose=melrose, flat=flat)) .* c_km .* t .* (g_rr ./ erg_inf) ./ erg_inf;
        du[:, 1:3] .= grad(hamiltonian(xVals, seed(k_loc), time[1], -view(u, :, 7) .* sqrt.(g_rr), θm, ωPul, B0, rNS, Mass_NS, iso=isotropic, melrose=melrose, flat=flat)) .* c_km .* t .* (g_rr ./ erg_inf);
        
        # du[:, 7] .= -derivative(tI -> hamiltonian(xVals, k_loc, tI, -view(u, :, 7) .* sqrt.(g_rr), θm, ωPul, B0, rNS, Mass_NS, iso=isotropic, melrose=melrose, flat=flat), time[1])[:] .* t .* (g_rr[:] ./ (-view(u, :, 7) .* sqrt.(g_rr))) .* g_rr[:]; # note extra g_rr[:] is coming from fact that were doing E_inf
        
        du[:, 7] .= -hamiltonian(xVals, k_loc, Dual(time[1], 1.0), -view(u, :, 7) .* sqrt.(g_rr), θm, ωPul, B0, rNS, Mass_NS, iso=isotropic, melrose=melrose, flat=flat, time_deriv=true)[1].partials .* t .* (g_rr[:] ./ erg_inf);
        du[u[:,1] .<= rNS .* 1.01, :] .= 0.0;
    
        
    end
end




# propogate photon module
function propagate(x0::Matrix, k0::Matrix,  nsteps::Int, Mvars::Array, NumerP::Array; dt_minCut=1e-7)
    ln_tstart, ln_tend, ode_err = NumerP
    
    tspan = (ln_tstart, ln_tend)
    saveat = (tspan[2] .- tspan[1]) ./ (nsteps-1)
    
    θm, ωPul, B0, rNS, gammaF, time0, Mass_NS, Mass_a, erg_loc, erg_inf, flat, isotropic, melrose = Mvars;
    
    
    # Define the Schwarzschild radius of the NS (in km)
    r_s0 = 2.0 * Mass_NS * GNew / c_km^2

    # Switch to polar coordinates
    rr = sqrt.(sum(x0.^2, dims=2))
    # r theta phi
    x0_pl = [rr acos.(x0[:,3] ./ rr) atan.(x0[:,2], x0[:,1])]
    
    omP = func_Plasma(x0_pl, time0, θm, ωPul, B0, rNS, zeroIn=true);
    
    # vr, vtheta, vphi --- Define lower momenta and upper indx pos
    # [unitless, unitless, unitless ]
    dr_dt = sum(x0 .* k0, dims=2) ./ rr
    v0_pl = [dr_dt (x0[:,3] .* dr_dt .- rr .* k0[:,3]) ./ (rr .* sin.(x0_pl[:,2])) (-x0[:,2] .* k0[:,1] .+ x0[:,1] .* k0[:,2]) ./ (rr .* sin.(x0_pl[:,2])) ];
    
    # Switch to celerity in polar coordinates
    AA = (1.0 .- r_s0 ./ rr)
    
    w0_pl = [v0_pl[:,1] ./ sqrt.(AA)   v0_pl[:,2] ./ rr .* rr.^2  v0_pl[:,3] ./ (rr .* sin.(x0_pl[:,2])) .* (rr .* sin.(x0_pl[:,2])).^2 ] ./ AA # lower index defined, [eV, eV * km, eV * km]
    w0_pl .*= 1.0 ./ erg_inf # renormalized to remove erg
    
    
    # Define initial conditions so that u0[1] returns a list of x positions (again, 1 entry for each axion trajectory) etc.
    
    # test = hamiltonian(x0_pl, w0_pl .* erg_inf, time0, erg_inf, θm, ωPul, B0, rNS, Mass_NS; iso=isotropic, melrose=melrose, zeroIn=false) ./ erg.^2
    
    
    
    # u0 = ([x0_pl w0_pl -erg_inf zeros(size(erg_inf)) zeros(size(erg_inf)) ])
    u0 = ([x0_pl w0_pl -erg_inf])
   
   # print(erg, "\n\n\n")
    
    function out_domain(u, Mvars, lnt)
        k = u[:, 4:6]
        xsp = u[:, 1:3]
        if flat
            g_tt, g_rr, g_thth, g_pp = g_schwartz(xsp, 0.0);
        else
            g_tt, g_rr, g_thth, g_pp = g_schwartz(xsp, Mass_NS);
        end
        testCond = (u[:,7] ./ sqrt.(g_rr) .- func_Plasma(u, exp.(lnt), θm, ωPul, B0, rNS)) ./ u[:,7]
        
        check = (testCond .<= 0.0)
        if sum(check) .> 0.0
            return true
        else
            return false
        end
    end
    condition_r(u,lnt,integrator) = sum(u[:, 1] .< (rNS*1.01)) .> 0
    affect_r!(integrator) = terminate!(integrator)
    # affect_r!(integrator) = u[u[:, 1] .< (rNS*1.01), 1] .= rNS;
    cb_r = DiscreteCallback(condition_r, affect_r!)
    condition_dt(u,lnt,integrator) = integrator.dt < dt_minCut
    #condition_dt(u,lnt,integrator) = integrator.dt < 1e-10
    function affect_dt!(integrator)
        if sum(integrator.u[:, 1] .<= 50.0 .* rNS) .> 0
            print("Time step too small, terminating \n")
            print(integrator.u, "\n")
            integrator.u[:, 1] .= rNS; # make sure run is killed
            terminate!(integrator)
        end
    end
    
    cb_dt = DiscreteCallback(condition_dt, affect_dt!)
    
    function condition_tau(u, lnt, integrator)
        r_s0 = 2.0 * Mass_NS * GNew / c_km^2
        AA = sqrt.(1.0 .- r_s0 ./ u[:, 1])
        locE = -u[:,7] ./ AA
       
        Bsphere = func_BField(u[:, 1:3], exp.(lnt), θm, ωPul, B0, rNS; Mass_NS=Mass_NS, flat=flat, sphericalX=true) .* 1.95e-2
        
        Bmag = sqrt.(spatial_dot(Bsphere, Bsphere, length(u[:, 1]), u[:, 1:3], Mass_NS));
        cyclF = 0.3 * Bmag / 5.11e5
        out = (cyclF .- locE) ./ cyclF
        return out[1]
    end
    
    tauVals = []
    function affect_tau!(integrator)
        omP = func_Plasma(integrator.u[:,1:3], exp.(integrator.t), θm, ωPul, B0, rNS, zeroIn=true);
        tau = pi / 3.0 .* (omP.^2 ./ -integrator.u[7]) .* integrator.u[1] ./ c_km ./ hbar
        push!(tauVals, tau[1])
    end
    
    
    # cb_tau = VectorContinuousCallback(condition_tau, affect_tau!, length(erg_inf),  interp_points=20, abstol=1e-4)
    cb_tau = ContinuousCallback(condition_tau, affect_tau!, interp_points=10, abstol=1e-2)
    
    cbset = CallbackSet(cb_r, cb_dt, cb_tau)
    
    # Define the ODEproblem
    Mvars_2 = θm, ωPul, B0, rNS, gammaF, time0, Mass_NS, Mass_a, erg_loc, erg_inf, flat, isotropic, melrose, w0_pl .* erg_inf
    prob = ODEProblem(func!, u0, tspan, Mvars_2, reltol=1e-5, dtmin=1e-9, maxiters=1e5, force_dtmin=true, abstol=ode_err, isoutofdomain=out_domain, callback=cbset)
    
    # Solve the ODEproblem
    sol = solve(prob, Vern6(), saveat=saveat)

    if (sol.retcode != :Success)&&(sol.retcode != :Terminated)
        r_s0 = 2.0 * Mass_NS * GNew / c_km^2
        AA = sqrt.(1.0 .- r_s0 ./ x0_pl[:, 1])
        print("problem? \n", x0, "\n", k0, "\n ", omP, "\t", erg_inf ./ AA, "\n\n")
        print(sol.u, "\t")
    end
    
    for i in 1:length(sol.u)
        sol.u[i][:,4:6] .*= erg_inf
    end
   
    
    # Define the Schwarzschild radii (in km)
    r_s = 2.0 .* ones(length(sol.u[1][:,1]), length(sol.u)) .* Mass_NS .* GNew ./ c_km^2
    

    # Calculate the total particle energies (unitless); this is later used to find the resonance and should be constant along the trajectory
    for i in 1:length(sol.u)
        sol.u[i][sol.u[i][:,1] .<= r_s[:,i], 1] .= 2.0 .* Mass_NS .* GNew ./ c_km^2 .+ 1e-10
    end
    ω = [(1.0 .- r_s[:,i] ./ sol.u[i][:,1]) for i in 1:length(sol.u)]



    # Switch back to proper velocity
    v_pl = [[sol.u[i][:,4] .* sqrt.(ω[i])  sol.u[i][:,5] ./ sol.u[i][:,1] sol.u[i][:,6] ./ (sol.u[i][:,1] .* sin.(sol.u[i][:,2])) ] .* ω[i] for i in 1:length(sol.u)]
    
    # Switch back to Cartesian coordinates
    x = [[sol.u[i][:,1] .* sin.(sol.u[i][:,2]) .* cos.(sol.u[i][:,3])  sol.u[i][:,1] .* sin.(sol.u[i][:,2]) .* sin.(sol.u[i][:,3])  sol.u[i][:,1] .* cos.(sol.u[i][:,2])] for i in 1:length(sol.u)]

    
    v = [[cos.(sol.u[i][:,3]) .* (sin.(sol.u[i][:,2]) .* v_pl[i][:,1] .+ cos.(sol.u[i][:,2]) .* v_pl[i][:,2]) .- sin.(sol.u[i][:,2]) .* sin.(sol.u[i][:,3]) .* v_pl[i][:,3] ./ sin.(sol.u[i][:,2]) sin.(sol.u[i][:,3]) .* (sin.(sol.u[i][:,2]) .* v_pl[i][:,1] .+ cos.(sol.u[i][:,2]) .* v_pl[i][:,2]) .+  sin.(sol.u[i][:,2]) .* cos.(sol.u[i][:,3]) .* v_pl[i][:,3] ./ sin.(sol.u[i][:,2]) cos.(sol.u[i][:,2]) .* v_pl[i][:,1] .-  sin.(sol.u[i][:,2]) .* v_pl[i][:,2] ] for i in 1:length(sol.u)]
    
    
    # Define the return values so that x_reshaped, v_reshaped (called with propagateAxion()[1] and propagateAxion()[2] respectively) are st by 3 by nsteps arrays (3 coordinates at different timesteps for different axion trajectories)
    
    # second one is index up, first is index down
    dxdtau = [[sol.u[i][:,4]  sol.u[i][:,5]  sol.u[i][:,6] ] for i in 1:length(sol.u)]
    # dxdtau = [[sol.u[i][:,4] .* ω[i] sol.u[i][:,5] ./ sol.u[i][:,1].^2 sol.u[i][:,6] ./ (sol.u[i][:,1] .*  sin.(sol.u[i][:,2])).^2 ] for i in 1:length(sol.u)]
    
    
    x_reshaped = cat([x[:, 1:3] for x in x]..., dims=3)
    v_reshaped = cat([v[:, 1:3] for v in v]..., dims=3)
    dxdtau = cat([dxdtau[:, 1:3] for dxdtau in dxdtau]..., dims=3)
    sphere_c = cat([sol.u[i][:, 1:3] for i in 1:length(sol.u)]..., dims=3)
    
    
    dt = cat([Array(u)[:, 7] for u in sol.u]..., dims = 2);
    
    
    fail_indx = ones(length(sphere_c[:, 1, end]))
    fail_indx[sphere_c[:, 1, end] .<= (rNS .* 1.01)] .= 0.0
    
   
    times = sol.t
    if length(tauVals) == 0
        tauVals = [0.0]
    end
    
    return x_reshaped, v_reshaped, dt, fail_indx, tauVals[1]
    
end


function g_schwartz(x0, Mass_NS; rNS=10.0, deriv=false, deriv_indx=0)
    # (1 - r_s / r)
    # notation (-,+,+,+), defined upper index g^mu^nu
    
    if length(size(x0)) > 1
        r = x0[:,1]
        rs = ones(eltype(r), size(r)) .* 2 * GNew .* Mass_NS ./ c_km.^2
        rs[r .<= rNS] .*= (r[r .<= rNS] ./ rNS).^3
        sin_theta = sin.(x0[:,2])
    else
        rs = 2 * GNew .* Mass_NS ./ c_km.^2
        r = x0[1]
        if r <= rNS
            rs .*= (r ./ rNS).^3
        end
        sin_theta = sin.(x0[2])
    end
    
    if !deriv
        g_tt = -1.0 ./ (1.0 .- rs ./ r);
        g_rr = (1.0 .- rs ./ r);
        g_thth = 1.0 ./ r.^2; # 1/km^2
        g_pp = 1.0 ./ (r.^2 .* sin_theta.^2); # 1/km^2
        return g_tt, g_rr, g_thth, g_pp
    else
        if deriv_indx == 0
            return -1.0 ./ (1.0 .- rs ./ r);
        elseif deriv_indx == 1
            return (1.0 .- rs ./ r);
        elseif deriv_indx == 2
            return 1.0 ./ r.^2;
        elseif deriv_indx == 3
            return 1.0 ./ (r.^2 .* sin_theta.^2);
        end
    end
    
end



function solve_vel_CS(θ, ϕ, r, NS_vel; guess=[0.1 0.1 0.1 1.0], errV=1e-20, Mass_NS=1.0)
    ff = sum(NS_vel.^2); # unitless
    
    GMr = GNew .* Mass_NS ./ r ./ (c_km .^ 2); # unitless
    rhat = [sin.(θ) .* cos.(ϕ) sin.(θ) .* sin.(ϕ) cos.(θ)]

    function f!(F, x)
        vx, vy, vz, Lam = x
        v_in = [vx vy vz]
        ff_loc = (sum(v_in.^2) .- (2 .* GMr .+ ff)) ./ ff
        # Lam = 1.0
        denom = ff .+ GMr .- sqrt.(ff) .* sum(v_in .* rhat);
        
        F[1] = abs.((ff .* vx .+ sqrt.(ff) .* GMr .* rhat[1] .- sqrt.(ff) .* vx .* sum(v_in .* rhat)) ./ (NS_vel[1] .* denom) .- 1.0) .+ Lam .* abs.(ff_loc)
        F[2] = abs.((ff .* vy .+ sqrt.(ff) .* GMr .* rhat[2] .- sqrt.(ff) .* vy .* sum(v_in .* rhat)) ./ (NS_vel[2] .* denom) .- 1.0) .+ Lam .* abs.(ff_loc)
        F[3] = abs.((ff .* vz .+ sqrt.(ff) .* GMr .* rhat[3] .- sqrt.(ff) .* vz .* sum(v_in .* rhat)) ./ (NS_vel[3] .* denom) .- 1.0) .+ Lam .* abs.(ff_loc)
        F[4] = abs.(ff_loc)
        
        
    end
  
    soln = nlsolve(f!, guess, autodiff = :forward, ftol=errV, iterations=10000)

    # FF = zeros(3)
    FF = zeros(4)
    f!(FF, soln.zero)
    accur = sqrt.(sum(FF.^2))
    # print("accuracy... ", FF, "\t", soln.zero, "\n")
    
    return soln.zero, accur
end

function jacobian_fv(x_in, vel_loc)

    rmag = sqrt.(sum(x_in.^2));
    ϕ = atan.(x_in[2], x_in[1])
    θ = acos.(x_in[3] ./ rmag)
    
    dvXi_dV = grad(v_infinity(θ, ϕ, rmag, seed(transpose(vel_loc)), v_comp=1));
    dvYi_dV = grad(v_infinity(θ, ϕ, rmag, seed(transpose(vel_loc)), v_comp=2));
    dvZi_dV = grad(v_infinity(θ, ϕ, rmag, seed(transpose(vel_loc)), v_comp=3));
    
    
    JJ = det([dvXi_dV; dvYi_dV; dvZi_dV])
    # print([dvXi_dV; dvYi_dV; dvZi_dV], "\n")
    return abs.(JJ).^(-1)
end

function v_infinity(θ, ϕ, r, vel_loc; v_comp=1, Mass_NS=1)
    vx, vy, vz = vel_loc
    vel_loc_mag = sqrt.(sum(vel_loc.^2))
    GMr = GNew .* Mass_NS ./ r ./ (c_km .^ 2); # unitless

    v_inf = sqrt.(vel_loc_mag.^2 .- (2 .* GMr)); # unitless
    rhat = [sin.(θ) .* cos.(ϕ) sin.(θ) .* sin.(ϕ) cos.(θ)]
    r_dot_v = sum(vel_loc .* rhat)
    
    
    denom = v_inf.^2 .+ GMr .- v_inf .* r_dot_v;
    
    if v_comp == 1
        v_inf_comp = (v_inf.^2 .* vx .+ v_inf .* GMr .* rhat[1] .- v_inf .* vx .* r_dot_v) ./ denom
    elseif v_comp == 2
        v_inf_comp = (v_inf.^2 .* vy .+ v_inf .* GMr .* rhat[2] .- v_inf .* vy .* r_dot_v) ./ denom
    elseif v_comp == 3
        v_inf_comp = (v_inf.^2 .* vz .+ v_inf .* GMr .* rhat[3] .- v_inf .* vz .* r_dot_v) ./ denom
    else
        vxC = (v_inf.^2 .* vx .+ v_inf .* GMr .* rhat[1] .- v_inf .* vx .* r_dot_v) ./ denom
        vyC = (v_inf.^2 .* vy .+ v_inf .* GMr .* rhat[2] .- v_inf .* vy .* r_dot_v) ./ denom
        vzC = (v_inf.^2 .* vz .+ v_inf .* GMr .* rhat[3] .- v_inf .* vz .* r_dot_v) ./ denom
        v_inf_comp = [vxC vyC vzC]
    end
    return v_inf_comp
end



function g_det(x0, t, θm, ωPul, B0, rNS, Mass_NS; flat=false)
    # returns determinant of sqrt(-g)
    if flat
        return ones(length(x0[:, 1]))
    end
    
    g_tt, g_rr, g_thth, g_pp = g_schwartz(x0, Mass_NS; rNS=rNS);
    
    r = x0[:,1]
    dωp = grad(func_Plasma(seed(x0), t, θm, ωPul, B0, rNS, zeroIn=false));
    
    dωt = func_Plasma(x0, Dual(t[1], 1.0), θm, ωPul, B0, rNS, zeroIn=false, time_deriv=true)[1].partials;
    
    dr_t = dωp[:, 1].^(-1) .* dωt ./ c_km # unitless
    dr_th = dωp[:, 1].^(-1) .* dωp[:, 2] # km
    dr_p = dωp[:, 1].^(-1) .* dωp[:, 3] # km
    A = g_rr
    
    sqrt_det_g = r .* sqrt.(sin.(x0[:,2]).^2 .* (A .* r.^2 .+ dr_th.^2) .+ dr_p.^2)
    sqrt_det_g_noGR = r .* sqrt.(sin.(x0[:,2]).^2 .* (r.^2 .+ dr_th.^2) .+ dr_p.^2)
    return sqrt_det_g ./ sqrt_det_g_noGR # km^2,
end

function test_on_shell(x, v_loc, vIfty_mag, time0, θm, ωPul, B0, rNS, Mass_NS, Mass_a; iso=true, melrose=false, printStuff=false)
    # pass cartesian form
    # Define the Schwarzschild radius of the NS (in km)
    r_s0 = 2.0 * Mass_NS * GNew / c_km^2

    # Switch to polar coordinates
    rr = sqrt.(sum(x.^2, dims=2))
    # r theta phi
    x0_pl = [rr acos.(x[:,3] ./ rr) atan.(x[:,2], x[:,1])]
    
    AA = (1.0 .- r_s0 ./ rr)
    AA[rr .< rNS] .= (1.0 .- r_s0 ./ rNS)
    
    gammaA = 1 ./ sqrt.(1.0 .- (vIfty_mag ./ c_km).^2 )
    erg_inf = Mass_a .* sqrt.(1 .+ (vIfty_mag ./ c_km .* gammaA).^2)
    erg_loc = erg_inf ./ sqrt.(AA)

    v0 = transpose(v_loc) .* (erg_loc ./ sqrt.(erg_loc.^2 .+ Mass_a.^2)) # actually momentum
    
    omP = func_Plasma(x0_pl, time0, θm, ωPul, B0, rNS, zeroIn=false);
    
    # vr, vtheta, vphi --- Define lower momenta and upper indx pos
    # [unitless, unitless, unitless ]
    dr_dt = sum(x .* v0, dims=2) ./ rr[:, 1]
    v0_pl = [dr_dt (x[:,3] .* dr_dt .- rr .* v0[:, 3]) ./ (rr .* sin.(x0_pl[:,2])) (-x[:,2] .* v0[:, 1] .+ x[:,1] .* v0[:, 2]) ./ (rr .* sin.(x0_pl[:,2])) ];
    # Switch to celerity in polar coordinates
    w0_pl = [v0_pl[:,1] ./ sqrt.(AA)   v0_pl[:,2] ./ rr .* rr.^2  v0_pl[:,3] ./ (rr .* sin.(x0_pl[:,2])) .* (rr .* sin.(x0_pl[:,2])).^2 ] ./ AA # lower index defined, [eV, eV * km, eV * km]
    g_tt, g_rr, g_thth, g_pp = g_schwartz(x0_pl, Mass_NS);

    NrmSq = (-erg_inf.^2 .* g_tt .- Mass_a.^2) ./ (w0_pl[:, 1].^2 .* g_rr .+  w0_pl[:, 2].^2 .* g_thth .+ w0_pl[:, 3].^2 .* g_pp )
    w0_pl .*= sqrt.(NrmSq)
    val = hamiltonian(x0_pl, w0_pl, time0, erg_inf, θm, ωPul, B0, rNS, Mass_NS; iso=iso, melrose=melrose, zeroIn=false) ./ erg_inf.^2
    
    tVals = erg_loc .> omP
    val_final = val[tVals[:]]
    min_value = minimum(abs.(val))
    return val_final, tVals[:], min_value
    
end

function hamiltonian(x, k,  time0, erg, θm, ωPul, B0, rNS, Mass_NS; iso=true, melrose=false, flat=false, zeroIn=false, time_deriv=false)
    # erg is energy at infinity
    # if r < rNS, need to not run...
    x[x[:,1] .< rNS, 1] .= rNS;

    
    omP = func_Plasma(x, time0, θm, ωPul, B0, rNS, zeroIn=zeroIn, time_deriv=time_deriv);
    
    # omP = func_Plasma(x, time0, θm, ωPul, B0, rNS, zeroIn=zeroIn);
    
    if flat
        g_tt, g_rr, g_thth, g_pp = g_schwartz(x, 0.0);
    else
        g_tt, g_rr, g_thth, g_pp = g_schwartz(x, Mass_NS);
    end
    ksqr = 0.0;
    try
        ksqr = g_tt .* erg.^2 .+ g_rr .* k[:, 1].^2 .+ g_thth .* k[:, 2].^2 .+ g_pp .* k[:, 3].^2
    catch
        ksqr = g_tt .* erg.^2 .+ g_rr .* k[1].^2 .+ g_thth .* k[2].^2 .+ g_pp .* k[3].^2
    end
    
    if iso
        Ham = 0.5 .* (ksqr .+ omP.^2)
    else
        if !melrose
            if flat
                ctheta = Ctheta_B_sphere(x, k, [θm, ωPul, B0, rNS, time0, 0.0])
            else
                ctheta = Ctheta_B_sphere(x, k, [θm, ωPul, B0, rNS, time0, Mass_NS])
            end
            Ham = 0.5 .* (ksqr .- omP.^2 .* (1.0 .- ctheta.^2) ./ (omP.^2 .* ctheta.^2 .- erg.^2 ./ g_rr)  .* erg.^2 ./ g_rr) # original form
        else
            if flat
                kpar = K_par(x, k, [θm, ωPul, B0, rNS, time0, 0.0])
            else
                kpar = K_par(x, k, [θm, ωPul, B0, rNS, time0, Mass_NS])
            end
            Ham = 0.5 .* (ksqr .+ omP.^2 .* (erg.^2 ./ g_rr .- kpar.^2) ./  (erg.^2 ./ g_rr)  );
        end
    end
    
    return Ham
end

function omega_function(x, k, time0, erg, θm, ωPul, B0, rNS, Mass_NS; kmag=nothing, cthetaB=nothing, iso=true, melrose=false, flat=false, zeroIn=false)
    # if r < rNS, need to not run...
    
    x[x[:,1] .< rNS, 1] .= rNS;
    
    omP = func_Plasma(x, time0, θm, ωPul, B0, rNS, zeroIn=zeroIn);
    if flat
        g_tt, g_rr, g_thth, g_pp = g_schwartz(x, 0.0);
    else
        g_tt, g_rr, g_thth, g_pp = g_schwartz(x, Mass_NS);
    end
    ksqr = 0.0;
    

    if isnothing(kmag)
        try
            ksqr = g_rr .* k[:, 1].^2 .+ g_thth .* k[:, 2].^2 .+ g_pp .* k[:, 3].^2
        catch
            ksqr = g_rr .* k[1].^2 .+ g_thth .* k[2].^2 .+ g_pp .* k[3].^2
        end
        
    else
        ksqr = kmag.^2
    end

    
    if iso
        Ham = (ksqr .+ omP.^2)
    else
        if isnothing(kmag)
            if flat
                kpar = K_par(x, k, [θm, ωPul, B0, rNS, time0, 0.0])
            else
                kpar = K_par(x, k, [θm, ωPul, B0, rNS, time0, Mass_NS])
            end
        else
            kpar = kmag .* cthetaB
        end
#         kpar = K_par(x, k, [θm, ωPul, B0, rNS, time0, Mass_NS])
        Ham = (ksqr .+ omP.^2 .+ sqrt.(ksqr.^2 .+ 2 .* ksqr .* omP.^2 .- 4 .* kpar.^2 .* omP.^2 .+ omP.^4))
    end

    return sqrt.(Ham)
end


function k_sphere(x0, k0, θm, ωPul, B0, rNS, time0, Mass_NS, flat; zeroIn=true, spherical=false)
    if flat
        Mass_NS = 0.0;
    end
    
    # Define the Schwarzschild radius of the NS (in km)
    r_s0 = 2.0 * Mass_NS * GNew / c_km^2

    
    # Switch to polar coordinates
    rr = sqrt.(sum(x0.^2, dims=2))
    # r theta phi
    x0_pl = [rr acos.(x0[:,3] ./ rr) atan.(x0[:,2], x0[:,1])]
    
    omP = func_Plasma(x0_pl, time0, θm, ωPul, B0, rNS, zeroIn=zeroIn, sphericalX=true);
    
    # vr, vtheta, vphi --- Define lower momenta and upper indx pos
    # [unitless, unitless, unitless ]
    
    dr_dt = sum(x0 .* k0, dims=2) ./ rr
    v0_pl = [dr_dt (x0[:,3] .* dr_dt .- rr .* k0[:,3]) ./ (rr .* sin.(x0_pl[:,2])) (-x0[:,2] .* k0[:,1] .+ x0[:,1] .* k0[:,2]) ./ (rr .* sin.(x0_pl[:,2])) ];
    
    # Switch to celerity in polar coordinates
    AA = (1.0 .- r_s0 ./ rr)
    
    w0_pl = [v0_pl[:,1] ./ sqrt.(AA)   v0_pl[:,2] ./ rr .* rr.^2  v0_pl[:,3] ./ (rr .* sin.(x0_pl[:,2])) .* (rr .* sin.(x0_pl[:,2])).^2 ] ./ AA # lower index defined, [eV, eV * km, eV * km]
    return w0_pl
end

function k_norm_Cart(x0, khat, time0, erg, θm, ωPul, B0, rNS, Mass_NS, Mass_a; melrose=false, flat=false, isotropic=false, ax_fix=true)

    # Switch to polar coordinates
    rr = sqrt.(sum(x0.^2, dims=2))
    # r theta phi
    x0_pl = [rr acos.(x0[:,3] ./ rr) atan.(x0[:,2], x0[:,1])]

    r_s0 = 2.0 * Mass_NS * GNew / c_km^2
    
    # vr, vtheta, vphi --- Define lower momenta and upper indx pos
    # [unitless, unitless, unitless ]
    dr_dt = sum(x0 .* khat, dims=2) ./ rr
    v0_pl = [dr_dt (x0[:,3] .* dr_dt .- rr .* khat[:,3]) ./ (rr .* sin.(x0_pl[:,2])) (-x0[:,2] .* khat[:,1] .+ x0[:,1] .* khat[:,2]) ./ (rr .* sin.(x0_pl[:,2])) ];
    
    # Switch to celerity in polar coordinates
    AA = (1.0 .- r_s0 ./ rr)
    
    w0_pl = [v0_pl[:,1] ./ sqrt.(AA)   v0_pl[:,2] ./ rr .* rr.^2  v0_pl[:,3] ./ (rr .* sin.(x0_pl[:,2])) .* (rr .* sin.(x0_pl[:,2])).^2 ] ./ AA  # lower index defined, [eV, eV * km, eV * km]
    
    omP = func_Plasma(x0_pl, time0, θm, ωPul, B0, rNS);
    
    g_tt, g_rr, g_thth, g_pp = g_schwartz(x0_pl, Mass_NS);
    
    if ax_fix
        NrmSq = (-erg.^2 .* g_tt .- Mass_a.^2) ./ (w0_pl[:, 1].^2 .* g_rr .+  w0_pl[:, 2].^2 .* g_thth .+ w0_pl[:, 3].^2 .* g_pp )
    else
        if !isotropic
            kpar = K_par(x0_pl, w0_pl, [θm, ωPul, B0, rNS, time0, Mass_NS]; flat=flat)
        else
            kpar = 0.0
        end
        
        NrmSq = (-erg.^2 .* g_tt .- omP.^2) ./ (w0_pl[:, 1].^2 .* g_rr .+  w0_pl[:, 2].^2 .* g_thth .+ w0_pl[:, 3].^2 .* g_pp .- omP.^2 ./ (-erg.^2 .* g_tt) .* kpar.^2 )
    end

          
    return sqrt.(NrmSq) .* khat
    
end





function surfNorm(x0, k0, Mvars; return_cos=true)
    # coming in cartesian, so change.
    
    
    θm, ωPul, B0, rNS, gammaF, t_start, Mass_NS = Mvars
    
    
    rr = sqrt.(sum(x0.^2, dims=2))
    # r theta phi
    x0_pl = [rr acos.(x0[:,3] ./ rr) atan.(x0[:,2], x0[:,1])]
    # vr, vtheta, vphi --- Define lower momenta and upper indx pos
    # [unitless, unitless, unitless ]
    dr_dt = sum(x0 .* k0, dims=2) ./ rr
    v0_pl = [dr_dt (x0[:,3] .* dr_dt .- rr .* k0[:,3]) ./ (rr .* sin.(x0_pl[:,2])) (-x0[:,2] .* k0[:,1] .+ x0[:,1] .* k0[:,2]) ./ (rr .* sin.(x0_pl[:,2])) ];
    # Switch to celerity in polar coordinates
    r_s0 = 2.0 * Mass_NS * GNew / c_km^2
    AA = (1.0 .- r_s0 ./ rr)
    w0_pl = [v0_pl[:,1] ./ sqrt.(AA)   v0_pl[:,2] ./ rr .* rr.^2  v0_pl[:,3] ./ (rr .* sin.(x0_pl[:,2])) .* (rr .* sin.(x0_pl[:,2])).^2 ] ./ AA # lower index defined, [eV, eV * km, eV * km]
    g_tt, g_rr, g_thth, g_pp = g_schwartz(x0_pl, Mass_NS)
    
    dωdr_grd = grad(func_Plasma(seed(x0_pl), t_start, θm, ωPul, B0, rNS))
    snorm = dωdr_grd ./ sqrt.(g_rr .* dωdr_grd[:, 1].^2  .+ g_thth .* dωdr_grd[:, 2].^2 .+ g_pp .* dωdr_grd[:, 3].^2)
    knorm = sqrt.(g_rr .* w0_pl[:,1].^2 .+ g_thth .* w0_pl[:,2].^2 .+ g_pp .* w0_pl[:,3].^2)
    ctheta = (g_rr .* w0_pl[:,1] .* snorm[:, 1] .+ g_thth .* w0_pl[:,2] .* snorm[:, 2] .+ g_pp .* w0_pl[:,3] .* snorm[:, 3]) ./ knorm
    
    
    if return_cos
        return ctheta
    else
        return ctheta, snorm
    end
end

function angle_vg_sNorm(x0, k0, thetaB, Mvars; return_cos=true)
    # coming in cartesian, so change.
        
    θm, ωPul, B0, rNS, gammaF, t_start, Mass_NS, flat, isotropic, erg = Mvars

    rr = sqrt.(sum(x0.^2, dims=2))
    # r theta phi
    x0_pl = [rr acos.(x0[:,3] ./ rr) atan.(x0[:,2], x0[:,1])]
    # vr, vtheta, vphi --- Define lower momenta and upper indx pos
    # [unitless, unitless, unitless ]
    dr_dt = sum(x0 .* k0, dims=2) ./ rr
    v0_pl = [dr_dt (x0[:,3] .* dr_dt .- rr .* k0[:,3]) ./ (rr .* sin.(x0_pl[:,2])) (-x0[:,2] .* k0[:,1] .+ x0[:,1] .* k0[:,2]) ./ (rr .* sin.(x0_pl[:,2])) ];
    # Switch to celerity in polar coordinates
    r_s0 = 2.0 * Mass_NS * GNew / c_km^2
    AA = (1.0 .- r_s0 ./ rr)
    vg = [v0_pl[:,1] ./ sqrt.(AA)   v0_pl[:,2] ./ rr .* rr.^2  v0_pl[:,3] ./ (rr .* sin.(x0_pl[:,2])) .* (rr .* sin.(x0_pl[:,2])).^2]  ./ AA # lower index defined, [eV, eV * km, eV * km]
    
    g_tt, g_rr, g_thth, g_pp = g_schwartz(x0_pl, Mass_NS)
    dωdr_grd = grad(func_Plasma(seed(x0_pl), t_start, θm, ωPul, B0, rNS))
    snorm = dωdr_grd ./ sqrt.(g_rr .* dωdr_grd[:, 1].^2  .+ g_thth .* dωdr_grd[:, 2].^2 .+ g_pp .* dωdr_grd[:, 3].^2)
    vgNorm = sqrt.(g_rr .* vg[:,1].^2 .+ g_thth .* vg[:,2].^2 .+ g_pp .* vg[:,3].^2)
    
    omP = func_Plasma(x0_pl, t_start, θm, ωPul, B0, rNS)
    
    ctheta = (g_rr .* vg[:,1] .* snorm[:, 1] .+ g_thth .* vg[:,2] .* snorm[:, 2] .+ g_pp .* vg[:,3] .* snorm[:, 3]) ./ vgNorm
    
    if return_cos
        return ctheta
    else
        return ctheta, snorm
    end
end



function K_par(x0_pl, k_sphere, Mvars; flat=false)
    θm, ωPul, B0, rNS, t_start, Mass_NS = Mvars
    ntrajs = length(x0_pl[:, 1])
    Bsphere = func_BField(x0_pl, t_start, θm, ωPul, B0, rNS; Mass_NS=Mass_NS, flat=flat, sphericalX=true)
    Bmag = func_BField(x0_pl, t_start, θm, ωPul, B0, rNS; Mass_NS=Mass_NS, flat=flat, sphericalX=true, return_comp=0) ./ 1.95e-2
    
    g_tt, g_rr, g_thth, g_pp = g_schwartz(x0_pl, Mass_NS)
#    if size(x0_pl)[1] > 1
#        k_par = (g_rr .* k_sphere[:,1] .* Bsphere[:, 1] .+ g_thth .* k_sphere[:,2] .* Bsphere[:, 2] .+ g_pp .* k_sphere[:,3] .* Bsphere[:, 3])  ./ Bmag
#    else
#        k_par = (g_rr .* k_sphere[1] .* Bsphere[1] .+ g_thth .* k_sphere[2] .* Bsphere[2] .+ g_pp .* k_sphere[3] .* Bsphere[3])  ./ Bmag
#    end
    k_par = (g_rr[1] .* k_sphere[1] .* Bsphere[1] .+ g_thth[1] .* k_sphere[2] .* Bsphere[2] .+ g_pp[1] .* k_sphere[3] .* Bsphere[3])  ./ Bmag
    
    return k_par
end


function spherical_to_cartesian(xSphere)
    r = view(xSphere, :, 1)
    θ = view(xSphere, :, 2)
    ϕ = view(xSphere, :, 3)
    x = [r .* cos.(ϕ) .* sin.(θ) r .* sin.(ϕ) .* sin.(θ) r .* cos.(θ)]
    return x
end

function cartesian_to_spherical(xC)
    r = sqrt.(sum(xC.^2 , dims=2))
    ϕ = atan.(view(xC, :, 2), view(xC, :, 1))
    θ = acos.(view(xC, :, 3)./ r)
    
    x = [r θ ϕ]
    return x
end

function GJ_Model_ωp_vecSPH(x, t, θm, ω, B0, rNS; zeroIn=true, sphericalX=true)
    # For GJ model, return \omega_p [eV]
    # Assume \vec{x} is in Cartesian coordinates [km], origin at NS, z axis aligned with ω
    # theta_m angle between B field and rotation axis
    # t [s], omega [1/s]
    
#    xNew = zero(x)
#    for i in 1:length(x[:,1])
#        xNew[i, :] .= rotate_y(θR) * x[i, :]
#    end

    if !sphericalX
        if size(x)[1] > 1
            r = sqrt.(sum(x.^2 , dims=2))
            ϕ = atan.(view(x, :, 2), view(x, :, 1))
            θ = acos.(view(x, :, 3)./ r)
        else
            r = sqrt.(sum(x.^2))
            ϕ = atan.(x[2], x[1])
            θ = acos.(x[3] ./ r)
        end
    else
        if size(x)[1] > 1
            r = view(x, :, 1)
            θ = view(x, :, 2)
            ϕ = view(x, :, 3)
        else
            r = x[1]
            θ = x[2]
            ϕ = x[3]
        end
    end
    
    
    ψ = ϕ .- ω.*t
    Bnorm = B0 .* (rNS ./ r).^3 ./ 2
    
    Br = 2 .* Bnorm .* (cos.(θm) .* cos.(θ) .+ sin.(θm) .* sin.(θ) .* cos.(ψ))
    Btheta = Bnorm .* (cos.(θm) .* sin.(θ) .- sin.(θm) .* cos.(θ) .* cos.(ψ))
    Bphi = Bnorm .* sin.(θm) .* sin.(ψ)
    
    # Bx = Br .* sin.(θ) .* cos.(ϕ) .+ Btheta .* cos.(θ) .* cos.(ϕ) .- Bphi .* sin.(ϕ)
    # By = Br .* sin.(θ) .* sin.(ϕ) .+ Btheta .* cos.(θ) .* sin.(ϕ) .+ Bphi .* cos.(ϕ)
    Bz = Br .* cos.(θ) .- Btheta .* sin.(θ)
    
    nelec = abs.((2.0 .* ω .* Bz) ./ sqrt.(4 .* π ./ 137) .* 1.95e-2 .* hbar) ; # eV^3
    ωp = sqrt.(4 .* π .* nelec ./ 137 ./ 5.0e5);
    
    if zeroIn
        if size(ωp) == ()
            if r < rNS
                ωp = 0.0
            end
        else
            if sum(r .< rNS) > 0
                ωp[r .< rNS] .= 0.0;
            end
        end
    end
    return ωp
end

# roughly compute conversion surface, so we know where to efficiently sample
function Find_Conversion_Surface(Ax_mass, t_in, θm, ω, B0, rNS, gammaL, relativ, plasma_mod, funcB; dead=false, dead_rmax=30.0)

    rayT = RayTracerGR;
    if θm < (π ./ 2.0)
        θmEV = θm ./ 2.0
    else
        θmEV = (θm .+ π) ./ 2.0
    end
    
    ### THIS FUNCTION NEEDS TO BE GENERALIZED
    
    print(plasma_mod, "\n")
    if (plasma_mod == "GJ")||(plasma_mod == "SphereTest")
        # estimate max dist
        # om_test = func_Plasma(, t_in, θm, ω, B0, rNS; sphericalX=false, zeroIn=false);
        om_test = GJ_plasma([rNS 0.0 0.0], t_in, θm, ω, B0, rNS, funcB; zeroIn=false, sphericalX=true, null_fill=0.0, Mass_a=Ax_mass)
        
        rc_guess = rNS .* (om_test ./ Ax_mass) .^ (2.0 ./ 3.0);
        
    elseif plasma_mod == "Magnetar"
        om_test = func_Plasma(rNS .* [1.0 0.0 0.0], t_in, θm, ω, B0, rNS; sphericalX=false);
        rc_guess = rNS .* (om_test ./ Ax_mass) .^ (1.0 ./ 2.0);
    end
    
    if dead
        return dead_rmax + 5.0
    end
    
    return rc_guess .* 1.05 # add a bit just for buffer
end



function spatial_dot(vec1, vec2, ntrajs, x0_pl, Mass_NS)
    # assumes both vectors v_mu
    g_tt, g_rr, g_thth, g_pp = g_schwartz(x0_pl, Mass_NS);
    out_v = zeros(ntrajs)
    for i in 1:ntrajs
        out_v[i] = (g_rr[i] .* vec1[i, 1] .* vec2[i, 1] .+ g_thth[i] .* vec1[i, 2] .* vec2[i, 2] .+ g_pp[i] .* vec1[i, 3] .* vec2[i, 3])
    end
    return out_v
end


function k_diff(x0_pl, velNorm_flat, time0, erg_inf_ini, θm, ωPul, B0, rNS, Mass_NS, Mass_a; melrose=melrose, flat=flat, isotropic=isotropic)
    if flat
        Mass_NS = 0.0
    end
    ntrajs = zeros(length(erg_inf_ini))
    xpos_flat = [x0_pl[:,1] .* sin.(x0_pl[:,2]) .* cos.(x0_pl[:,3]) x0_pl[:,1] .* sin.(x0_pl[:,2]) .* sin.(x0_pl[:,3]) x0_pl[:,1] .* cos.(x0_pl[:,2])]
    
    k_init = k_norm_Cart(xpos_flat, velNorm_flat, 0.0, erg_inf_ini, θm, ωPul, B0, rNS, Mass_NS, Mass_a, melrose=melrose, flat=flat, isotropic=isotropic, ax_fix=false)
    ksphere = k_sphere(xpos_flat, k_init, θm, ωPul, B0, rNS, ntrajs, Mass_NS, flat, zeroIn=false)
    
    g_tt, g_rr, g_thth, g_pp = g_schwartz(x0_pl, Mass_NS);
    kmag = sqrt.(g_rr .* ksphere[:, 1].^2  .+ g_thth .* ksphere[:, 2].^2 .+ g_pp .* ksphere[:, 3].^2)
    erg_loc = erg_inf_ini ./ sqrt.(1.0 .- 2.0 .* GNew .* Mass_NS ./ x0_pl[:,1] ./ c_km.^2)
    k_ax = sqrt.(erg_loc.^2 .- Mass_a.^2)
    
    return (k_ax .- kmag)
    
    
end

function k_gamma(x0_pl, ksphere, time0, erg_inf_ini, θm, ωPul, B0, rNS, Mass_NS, Mass_a; melrose=false, flat=false, isotropic=false)
    g_tt, g_rr, g_thth, g_pp = g_schwartz(x0_pl, Mass_NS);
    
    Bsphere = func_BField(x0_pl, time0, θm, ωPul, B0, rNS; Mass_NS=Mass_NS, flat=flat, sphericalX=true)
    omP = func_Plasma(x0_pl, time0, θm, ωPul, B0, rNS, zeroIn=true)
    kmag = sqrt.(g_rr .* ksphere[:, 1].^2  .+ g_thth .* ksphere[:, 2].^2 .+ g_pp .* ksphere[:, 3].^2)
    Bmag = sqrt.(g_rr .* Bsphere[:, 1].^2  .+ g_thth .* Bsphere[:, 2].^2 .+ g_pp .* Bsphere[:, 3].^2)
    ctheta_B = (g_rr .* Bsphere[:, 1] .* ksphere[:, 1]  .+ g_thth .* Bsphere[:, 2] .* ksphere[:, 2] .+ g_pp .* Bsphere[:, 3] .* ksphere[:, 3])./ (kmag .* Bmag)
    if isotropic
        ctheta_B .*= 0.0
    end
    erg_loc = erg_inf_ini ./ sqrt.(g_rr)
    return erg_loc .* sqrt.(erg_loc.^2 .- omP.^2) ./  sqrt.(erg_loc.^2 .- omP.^2 .* ctheta_B.^2)
    
end


function Cristoffel(x0_pl, time0, θm, ωPul, B0, rNS, Mass_NS; flat=false)
    if flat
        MassNS = 0.0
    else
        MassNS = Mass_NS
    end
    r = x0_pl[1]
    theta = x0_pl[2]
    # note: t = theta, T = time
    GM = GNew .* Mass_NS ./ c_km.^2
    G_rrr = - GM ./ (r .* (r .- 2 .* GM))
    G_rtt = - (r - 2 .* GM)
    G_rpp = - (r - 2 .* GM) .* sin.(theta).^2
    G_trt = 1.0 ./ r
    G_tpp = -sin.(theta) .* cos.(theta)
    G_prp = 1.0 ./ r
    G_ptp = cos.(theta) ./ sin.(theta)
    
    G_ttr = 1.0 ./ r
    G_ppr = 1.0 ./ r
    G_ppt = cos.(theta) ./ sin.(theta)
    
    G_TTr = GM ./ (r .* (r - 2 .* GM))
    G_TrT = GM ./ (r .* (r - 2 .* GM))
    G_rTT = GM .* (r .- 2 .* GM) ./ r.^3
    
    return G_rrr, G_rtt, G_rpp, G_trt, G_tpp, G_prp, G_ptp, G_ttr, G_ppr, G_ppt, G_TTr, G_TrT, G_rTT
    # return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
end


function conversion_prob(Ax_g, x0_pl, Mvars, local_vars; one_D=false)
    
    # xIn cartesian, ksphere [spherical]
    θm, ωPul, B0, rNS, gammaF, t_start, Mass_NS, Mass_a, flat, isotropic, ωErg = Mvars
    ωp, Bsphere, Bmag, ksphere, kmag, vgNorm, ctheta_B, stheta_B = local_vars
    
    
    # TEST POINT
    # rs = 2 * GNew .* Mass_NS ./ c_km.^2
    # x0_pl = [19.579559672984 2.52058026435 1.739077233743]
    # ksphere = [2.97366458957e-6 6.266058430222e-5 1.2322604497409e-6]
    # g_tt, g_rr, g_thth, g_pp = g_schwartz(x0_pl, Mass_NS);
    # ωp = func_Plasma(x0_pl, t_start, θm, ωPul, B0, rNS, zeroIn=false);
    # Bsphere = func_BField(x0_pl, zeros(1), θm, ωPul, B0, rNS; Mass_NS=Mass_NS, flat=flat, sphericalX=true)
    # Bmag = sqrt.(spatial_dot(Bsphere, Bsphere, 1, x0_pl, Mass_NS)) .* 1.95e-2; # eV^2
    # kmag = sqrt.(spatial_dot(ksphere, ksphere, 1, x0_pl, Mass_NS));
    # ctheta_B = spatial_dot(Bsphere, ksphere, 1, x0_pl, Mass_NS) .* 1.95e-2 ./ (kmag .* Bmag)
    # stheta_B = sin.(acos.(ctheta_B))
    # erg_inf_ini = sqrt.(1.0 .- 2.0 .* GNew .* Mass_NS ./ x0_pl[:,1] ./ c_km.^2) .* ωErg
    # v_group = grad(omega_function(x0_pl, seed(ksphere), 0.0, -erg_inf_ini, θm, ωPul, B0, rNS, Mass_NS, flat=flat, iso=isotropic, melrose=true)) #
    # print(v_group, "\t", 1.0 ./ sqrt.(1 - rs ./ x0_pl[1]), "\n")
    # v_group[:, 1] ./= g_rr
    # v_group[:, 2] ./= g_thth
    # v_group[:, 3] ./= g_pp
    # vgNorm = sqrt.(spatial_dot(v_group, v_group, 1, x0_pl, Mass_NS));
    # print(x0_pl[:,1], "\t", ctheta_B, "\t", vgNorm, "\t", sqrt.(rs ./ x0_pl[:,1]), "\n")
    ####
    
    vloc = sqrt.(ωErg.^2 .- Mass_a.^2) ./ ωErg
    ntrajs = length(t_start)
    rr = x0_pl[:, 1]
    
    erg_inf_ini = sqrt.(1.0 .- 2.0 .* GNew .* Mass_NS ./ rr ./ c_km.^2) .* ωErg
    g_tt, g_rr, g_thth, g_pp = g_schwartz(x0_pl, Mass_NS);
    
    if isotropic
        dmu_E = grad(omega_function(seed(x0_pl), ksphere, t_start, erg_inf_ini, θm, ωPul, B0, rNS, Mass_NS, kmag=kmag, cthetaB=ctheta_B, iso=isotropic, flat=flat, melrose=true)) # based on energy gradient
        dmu_E_2 = dmu_E
        cos_wTEST = 0.0
    else
        G_rrr, G_rtt, G_rpp, G_trt, G_tpp, G_prp, G_ptp, G_ttr, G_ppr, G_ppt, G_TTr, G_TrT, G_rTT = Cristoffel(x0_pl, t_start, θm, ωPul, B0, rNS, Mass_NS; flat=flat)
        rs = 2 * GNew .* Mass_NS ./ c_km.^2
        
        dmu_omP = grad(func_Plasma(seed(x0_pl), t_start, θm, ωPul, B0, rNS, zeroIn=true));
        dmu_B = grad(func_BField(seed(x0_pl), t_start, θm, ωPul, B0, rNS; Mass_NS=Mass_NS, flat=flat, sphericalX=true, return_comp=0))
        
        
        # test
        grad_omega = grad(omega_function(seed(x0_pl), ksphere, t_start, erg_inf_ini, θm, ωPul, B0, rNS, Mass_NS, kmag=kmag, cthetaB=ctheta_B, iso=isotropic, flat=flat, melrose=true)) # based on energy gradient
        
        
        
        gradE_normTEST = grad_omega ./ sqrt.(g_rr .* grad_omega[:, 1].^2  .+ g_thth .* grad_omega[:, 2].^2 .+ g_pp .* grad_omega[:, 3].^2)
        cos_wTEST = abs.(spatial_dot(ksphere ./ kmag, gradE_normTEST, ntrajs, x0_pl, Mass_NS))
        
        
        term1 = ksphere[1] .* grad(func_BField(seed(x0_pl), t_start, θm, ωPul, B0, rNS; Mass_NS=Mass_NS, flat=flat, sphericalX=true, return_comp=1)) .+ ksphere[2] .* grad(func_BField(seed(x0_pl), t_start, θm, ωPul, B0, rNS; Mass_NS=Mass_NS, flat=flat, sphericalX=true, return_comp=2)) .+ ksphere[3] .* grad(func_BField(seed(x0_pl), t_start, θm, ωPul, B0, rNS; Mass_NS=Mass_NS, flat=flat, sphericalX=true, return_comp=3))
        
        term2_r = ksphere[1] .* (g_rr .* Bsphere[1] .* 1.95e-2) .* G_rrr .+ ksphere[2] .* G_trt .* (Bsphere[2] .* g_thth .* 1.95e-2) .+ ksphere[3] .* G_prp .* (Bsphere[3] .* g_pp .* 1.95e-2)
        term2_t = ksphere[1] .* (g_thth .* Bsphere[2] .* 1.95e-2) .* G_rtt .+ ksphere[3] .* G_ptp .* (Bsphere[3] .* g_pp .* 1.95e-2) .+ ksphere[2] .* (g_rr .* Bsphere[1] .* 1.95e-2) .* G_ttr
        term2_p =  ksphere[1] .* (g_pp .* Bsphere[3] .* 1.95e-2) .* G_rpp .+ ksphere[2] .* G_tpp .* (Bsphere[3] .* g_pp .* 1.95e-2) .+ ksphere[3] .* G_ppr .* (Bsphere[1] .* g_rr .* 1.95e-2) .+ ksphere[3] .* G_ppt .* (Bsphere[2] .* g_thth .* 1.95e-2)
        
        
        dmu_ctheta = (term1 .+ [term2_r term2_t term2_p]) ./ (kmag .* Bmag) .- ctheta_B .* dmu_B ./ Bmag
        
        v_group = grad(omega_function(x0_pl, seed(ksphere), t_start, -erg_inf_ini, θm, ωPul, B0, rNS, Mass_NS, flat=flat, iso=isotropic, melrose=true)) #
        term2_r = G_rrr .* ksphere[1] .* (g_rr .* v_group[1]) .+ G_trt .* ksphere[2] .* (g_thth .* v_group[2]) .+ G_prp .* ksphere[3] .* (g_pp .* v_group[3]) .+ G_TrT .* ωErg ./ sqrt.(1 .- rs ./ rr)
        term2_t = G_rtt .* ksphere[1] .* (g_thth .* v_group[2]) .+ G_ptp .* ksphere[3] .* (g_pp .* v_group[3]) .+ G_ttr .* ksphere[2] .* (g_rr .* v_group[1])
        term2_p = G_rpp .* ksphere[1] .* (g_pp .* v_group[3]) .+ G_tpp .* ksphere[2] .* (g_pp .* v_group[3]) .+ G_ppr .* ksphere[3] .* (g_rr .* v_group[1]) .+ G_ppt .* ksphere[3] .* (g_thth .* v_group[2])
        
        term2 = [term2_r term2_t term2_p]
       
        
        preF = ωp ./ abs.(ωErg.^5 .+ ctheta_B.^2 .* ωErg .* (ωp.^4 .- 2 .* ωp.^2 .* ωErg.^2))
        dmu_E = preF .* (ωErg.^4 .* stheta_B.^2 .* dmu_omP .- ωErg.^2 .* ctheta_B .* ωp .* (ωErg.^2 .- ωp.^2) .* dmu_ctheta)
        
        
        dmu_E_2 = dmu_E .+ term2
        vhat_gradE_2 = spatial_dot(ksphere ./ kmag, dmu_E_2, ntrajs, x0_pl, Mass_NS)
        
        final_part = g_tt .* erg_inf_ini .* (G_TTr .* erg_inf_ini .+ G_rTT .* ksphere[1]) .* sqrt.(-g_tt) ./ kmag
        
        vhat_gradE_2 .+= final_part
    end
    
    gradE_norm = dmu_E ./ sqrt.(g_rr .* dmu_E[:, 1].^2  .+ g_thth .* dmu_E[:, 2].^2 .+ g_pp .* dmu_E[:, 3].^2)
    gradE_norm_2 = dmu_E_2 ./ sqrt.(g_rr .* dmu_E_2[:, 1].^2  .+ g_thth .* dmu_E_2[:, 2].^2 .+ g_pp .* dmu_E_2[:, 3].^2)
    cos_w = abs.(spatial_dot(ksphere ./ kmag, gradE_norm, ntrajs, x0_pl, Mass_NS))
    
    
    cos_w_2 = abs.(spatial_dot(ksphere ./ kmag, gradE_norm_2, ntrajs, x0_pl, Mass_NS))
    vhat_gradE = spatial_dot(ksphere ./ kmag, dmu_E, ntrajs, x0_pl, Mass_NS)
    grad_Emag = spatial_dot(dmu_E, dmu_E, ntrajs, x0_pl, Mass_NS)
    grad_Emag_2 = spatial_dot(dmu_E_2, dmu_E_2, ntrajs, x0_pl, Mass_NS)
    
    if one_D
        Prob = π ./ 2.0 .* (Ax_g .* 1e-9 .* Bmag).^2  ./ (vloc .* (abs.(vhat_gradE) .* c_km .* hbar)) #
    else
        prefactor = ωErg.^4 .* stheta_B.^2 ./ (ctheta_B.^2 .* ωp.^2 .* (ωp.^2 .- 2 .* ωErg .^2) .+ ωErg.^4)
        Prob = π ./ 2.0 .* prefactor .* (Ax_g .* 1e-9 .* Bmag).^2 ./ (abs.(vhat_gradE) .* vloc .* c_km .* hbar)  # jamies paper
    end
    
    return Prob, abs.(vhat_gradE), abs.(cos_w), sqrt.(grad_Emag), abs.(cos_w_2), sqrt.(grad_Emag_2), abs.(cos_wTEST)
end

function dwp_ds(xIn, ksphere, Mvars)
    # xIn cartesian, ksphere [spherical]
    θm, ωPul, B0, rNS, gammaF, t_start, Mass_NS, Mass_a, flat, isotropic, ωErg = Mvars

    
    rr = sqrt.(sum(xIn.^2, dims=2))
    x0_pl = [rr acos.(xIn[:,3] ./ rr) atan.(xIn[:,2], xIn[:,1])]
    
    ntrajs = length(t_start)
    # general info we need for all
    omP = func_Plasma(x0_pl, t_start, θm, ωPul, B0, rNS, zeroIn=true)
    erg_inf_ini = sqrt.(1.0 .- 2.0 .* GNew .* Mass_NS ./ rr ./ c_km.^2) .* ωErg
    g_tt, g_rr, g_thth, g_pp = g_schwartz(x0_pl, Mass_NS);
    Bsphere = func_BField(xIn, t_start, θm, ωPul, B0, rNS; Mass_NS=Mass_NS, flat=flat)
    Bmag = sqrt.(spatial_dot(Bsphere, Bsphere, ntrajs, x0_pl, Mass_NS));
    kmag = sqrt.(spatial_dot(ksphere, ksphere, ntrajs, x0_pl, Mass_NS));
    kB_norm = spatial_dot(Bsphere ./ Bmag, ksphere ./ kmag, ntrajs, x0_pl, Mass_NS)
    v_ortho = -(Bsphere ./ Bmag .- kB_norm .* ksphere ./ kmag)
    v_ortho ./= sqrt.(spatial_dot(v_ortho, v_ortho, ntrajs, x0_pl, Mass_NS))
    
    ctheta_B = spatial_dot(Bsphere ./ Bmag, ksphere ./ kmag, ntrajs, x0_pl, Mass_NS)
    stheta_B = sin.(acos.(ctheta_B))
    if isotropic
        ctheta_B .*= 0.0
        stheta_B ./= stheta_B
    end
    xi = stheta_B .^2 ./ (1.0 .- ctheta_B.^2 .* omP.^2 ./ ωErg.^2)
    
    # omega_p computation
    grad_omP = grad(func_Plasma(seed(x0_pl), t_start, θm, ωPul, B0, rNS, zeroIn=true));
    dz_omP = spatial_dot(ksphere ./ kmag, grad_omP, ntrajs, x0_pl, Mass_NS)
    dy_omP = spatial_dot(v_ortho, grad_omP, ntrajs, x0_pl, Mass_NS)
    w_prime = dz_omP .+ omP.^2 ./ ωErg.^2 .* xi .* ctheta_B ./ stheta_B .* dy_omP
    
    # k gamma computation
    grad_kgamma = grad(k_gamma(seed(x0_pl), ksphere, t_start, erg_inf_ini, θm, ωPul, B0, rNS, Mass_NS, Mass_a; melrose=true, flat=flat, isotropic=isotropic))
    dz_k = spatial_dot(ksphere ./ kmag, grad_kgamma, ntrajs, x0_pl, Mass_NS)
    dy_k = spatial_dot(v_ortho, grad_kgamma, ntrajs, x0_pl, Mass_NS)
    # k_prime = dz_k .+ omP.^2 ./ ωErg.^2 .* xi .* ctheta_B ./ stheta_B .* dy_k
    # k_prime = dz_k .+ ctheta_B .* Mass_a.^2 ./ (stheta_B .* ωErg.^2) .* dy_k
    k_prime = dz_k .+ dy_k .* (omP.^2 ./ ωErg.^2) .* stheta_B .* ctheta_B ./ (-1 .+ omP.^2 .* ctheta_B.^2)
    gradK_norm = grad_kgamma ./ sqrt.(g_rr .* grad_kgamma[:, 1].^2  .+ g_thth .* grad_kgamma[:, 2].^2 .+ g_pp .* grad_kgamma[:, 3].^2)
    cos_k = abs.(spatial_dot(ksphere ./ kmag, gradK_norm, ntrajs, x0_pl, Mass_NS))
    
    # energy computation
    grad_omega = grad(omega_function(seed(x0_pl), ksphere, t_start, erg_inf_ini, θm, ωPul, B0, rNS, Mass_NS, kmag=kmag, cthetaB=ctheta_B, iso=isotropic,flat=flat, melrose=true)) # based on energy gradient

    erg_prime = spatial_dot(ksphere ./ kmag, grad_omega, ntrajs, x0_pl, Mass_NS)
    dz_w = spatial_dot(ksphere ./ kmag, grad_omega, ntrajs, x0_pl, Mass_NS)
    dy_w = spatial_dot(v_ortho, grad_omega, ntrajs, x0_pl, Mass_NS)
    erg_prime = dz_w .+ omP.^2 ./ ωErg.^2 .* xi .* ctheta_B ./ stheta_B .* dy_w
    gradE_norm = grad_omega ./ sqrt.(g_rr .* grad_omega[:, 1].^2  .+ g_thth .* grad_omega[:, 2].^2 .+ g_pp .* grad_omega[:, 3].^2)
    cos_w = abs.(spatial_dot(ksphere ./ kmag, gradE_norm, ntrajs, x0_pl, Mass_NS))
    erg_loc = erg_inf_ini ./ sqrt.(g_rr)
    
    
    # group velocity
    v_group = grad(omega_function(x0_pl, seed(ksphere), t_start, -erg_inf_ini, θm, ωPul, B0, rNS, Mass_NS, flat=flat, iso=isotropic, melrose=true)) #
    v_group[:, 1] ./= g_rr
    v_group[:, 2] ./= g_thth
    v_group[:, 3] ./= g_pp
    vgNorm = sqrt.(spatial_dot(v_group, v_group, ntrajs, x0_pl, Mass_NS));
    # print(vgNorm, "\t", ctheta_B, "\n")
    
    dsK = spatial_dot(v_group ./ vgNorm, grad_kgamma, ntrajs, x0_pl, Mass_NS)
    axV = sqrt.(ωErg.^2 .- omP.^2) ./ ωErg
    # print(ksphere ./ Mass_a, "\t", v_group, "\n")
    # print(abs.(k_prime ) ./ (abs.(erg_prime) ) , "\t", abs.(dsK) ./ (abs.(k_prime ./ (1 .+ ctheta_B.^2 .* omP.^2 ./ ωErg.^2)) ),  "\t", ctheta_B, "\n\n")
    c_kvg = spatial_dot(v_group ./ vgNorm, ksphere ./ kmag, ntrajs, x0_pl, Mass_NS)
    
    slength = sqrt.(1.0 .+ (omP.^2 ./ ωErg.^2 .* stheta_B.^2 ./ (1.0 .- omP.^2 ./ ωErg.^2 .* ctheta_B.^2) .* (ctheta_B ./stheta_B)  ).^2)
    if isotropic
        slength ./= slength
    end
    newGuess = (slength ./ vgNorm) .* spatial_dot(ksphere ./ kmag, grad_omega, ntrajs, x0_pl, Mass_NS)
    # print(newGuess ./ dz_k, "\t", vgNorm, "\t", sqrt.(ωErg.^2 .- omP.^2) ./ ωErg , "\n")
    
    
    
    dk_vg = abs.(spatial_dot(v_group ./ vgNorm, gradK_norm, ntrajs, x0_pl, Mass_NS)) # this is group velocity photon to surface normal
    k_vg = abs.(spatial_dot(v_group ./ vgNorm, ksphere ./ kmag, ntrajs, x0_pl, Mass_NS))
    dE_vg = abs.(spatial_dot(v_group ./ vgNorm, gradE_norm, ntrajs, x0_pl, Mass_NS)) # this is group velocity photon to surface normal

    # return abs.(w_prime), abs.(k_prime), abs.(erg_prime), cos_w, vgNorm, dk_vg, dE_vg, k_vg
    return abs.(w_prime), abs.(k_prime), abs.(newGuess), cos_w, vgNorm, dk_vg, dE_vg, k_vg
end




function find_samples_new(maxR, θm, ωPul, B0, rNS, Mass_a, Mass_NS; n_max=6, batchsize=2, thick_surface=false, iso=false, melrose=false, pre_randomized=nothing, t0=0.0, flat=false, rand_cut=true, delta_on_v=true, vmean_ax=220.0)
    
    if isnothing(pre_randomized)
        # ~~~collecting random samples
        
        # randomly sample angles θ, ϕ, hit conv surf
        θi = acos.(1.0 .- 2.0 .* rand());
        ϕi = rand() .* 2π;
        
        # local velocity
        θi_loc = acos.(1.0 .- 2.0 .* rand());
        ϕi_loc = rand() .* 2π;
        
        # randomly sample x1 and x2 (rotated vectors in disk perpendicular to (r=1, θ, ϕ) with max radius R)
        ϕRND = rand() .* 2π;
        
        # radius on disk
        rRND = sqrt.(rand()) .* maxR; # standard flat sampling
        
        
        # ~~~ Done collecting random samples
    else
        
        θi = acos.(1.0 .- 2.0 .* pre_randomized[1]);
        ϕi = pre_randomized[2] .* 2π;
        
        # local velocity
        θi_loc = acos.(1.0 .- 2.0 .* pre_randomized[3]);
        ϕi_loc = pre_randomized[4] .* 2π;
        
        # randomly sample x1 and x2 (rotated vectors in disk perpendicular to (r=1, θ, ϕ) with max radius R)
        ϕRND = pre_randomized[5] .* 2π;
        
        # radius on disk
        rRND = sqrt.(pre_randomized[6]) .* maxR; # standard flat sampling
        
    end
    
    # disk direction
    vvec_all = [sin.(θi) .* cos.(ϕi) sin.(θi) .* sin.(ϕi) cos.(θi)];
    
    # vel direction
    vvec_loc = [sin.(θi_loc) .* cos.(ϕi_loc) sin.(θi_loc) .* sin.(ϕi_loc) cos.(θi_loc)];
    
    x1 = rRND .* cos.(ϕRND);
    x2 = rRND .* sin.(ϕRND);
    # rotate using Inv[EurlerMatrix(ϕi, θi, 0)] on vector (x1, x2, 0)
    x0_all= [x1 .* cos.(-ϕi) .* cos.(-θi) .+ x2 .* sin.(-ϕi) x2 .* cos.(-ϕi) .- x1 .* sin.(-ϕi) .* cos.(-θi) x1 .* sin.(-θi)];
    
    if delta_on_v
        vIfty = (vmean_ax .+ rand(batchsize, 3) .* 1.0e-5) ./ sqrt.(3);
    else
        vIfty = vmean_ax .* erfinv.( 2.0 .* rand(batchsize, 3) .- 1.0);
        # vIfty = erfinv.(2 .* rand(batchsize, 3) .- 1.0) .* vmean_ax .+ v_NS # km /s
    end
    
    
    vIfty_mag = sqrt.(sum(vIfty.^2, dims=2));
    
    gammaA = 1 ./ sqrt.(1.0 .- (vIfty_mag ./ c_km).^2 )
    erg_inf_ini = Mass_a .* sqrt.(1 .+ (vIfty_mag ./ c_km .* gammaA).^2)
    # Mass_NS = 1.0
    
    # print(x0_all, "\t", vvec_all, "\n")
    x0_all .+= vvec_all .* (-maxR .* 1.1)
    
    xc = []; yc = []; zc = []
    
    
    function condition(u, t, integrator)
        if !thick_surface
            return (func_Plasma(u, t0, θm, ωPul, B0, rNS; sphericalX=false) .- Mass_a)[1] ./ Mass_a
        else
            
            r_s0 = 2.0 * Mass_NS * GNew / c_km^2
            rr = sqrt.(sum(u.^2))
            x0_pl = [rr acos.(u[3] ./ rr) atan.(u[2], u[1])]
            AA = (1.0 .- r_s0 ./ rr)
            if rr .< rNS
                AA = 1.0
            end
            
  
            dr_dt = sum(u .* vvec_loc) ./ rr
            v0_pl = [dr_dt (u[3] .* dr_dt .- rr .* vvec_loc[3]) ./ (rr .* sin.(x0_pl[2])) (-u[2] .* vvec_loc[1] .+ u[1] .* vvec_loc[2]) ./ (rr .* sin.(x0_pl[2])) ];
            # Switch to celerity in polar coordinates
            w0_pl = [v0_pl[1] ./ sqrt.(AA)   v0_pl[2] ./ rr .* rr.^2  v0_pl[3] ./ (rr .* sin.(x0_pl[2])) .* (rr .* sin.(x0_pl[2])).^2 ] ./ AA # lower index defined, [eV, eV * km, eV * km]
            g_tt, g_rr, g_thth, g_pp = g_schwartz(x0_pl, Mass_NS);

            NrmSq = (-erg_inf_ini.^2 .* g_tt .- Mass_a.^2) ./ (w0_pl[1].^2 .* g_rr .+  w0_pl[2].^2 .* g_thth .+ w0_pl[3].^2 .* g_pp )
            w0_pl .*= sqrt.(NrmSq)
            
            omP = func_Plasma(u, t0, θm, ωPul, B0, rNS; sphericalX=false)
            if iso
                kpar = 0.0
            else
                kpar = K_par(x0_pl, w0_pl, [θm, ωPul, B0, rNS, t0, Mass_NS])
            end
            ksqr = g_tt .* erg_inf_ini.^2 .+ g_rr .* w0_pl[1].^2 .+ g_thth .* w0_pl[2].^2 .+ g_pp .* w0_pl[3].^2
            Ham = 0.5 .* (ksqr .+ omP.^2 .* (erg_inf_ini.^2 ./ g_rr .- kpar.^2) ./  (erg_inf_ini.^2 ./ g_rr)  ) ./ (erg_inf_ini.^2);
            
            return Ham[1]
        end
    end
    
    function affect!(int)
        rr = sqrt.(sum(int.u.^2))
        x0_pl = [rr acos.(int.u[3] ./ rr) atan.(int.u[2], int.u[1])]
        omP = func_Plasma(int.u, t0, θm, ωPul, B0, rNS; sphericalX=false)[1]
        g_tt, g_rr, g_thth, g_pp = g_schwartz(x0_pl, Mass_NS);
        ergL = erg_inf_ini ./ sqrt.(g_rr)
        
        if (rr .> rNS) && (ergL[1] .> omP)
            push!( xc, int.u[1] )
            push!( yc, int.u[2]  )
            push!( zc, int.u[3]  )
        end
    end
    
    cb_s = ContinuousCallback(condition, affect!, interp_points=10, abstol=1e-4)
    
    function func_line!(du, u, Mvars, t)
        @inbounds begin
            kk, v_loc = Mvars
            du .= kk
        end
    end
    
    
    # solve differential equation with callback
    Mvars = [vvec_all, vvec_loc]
    prob = ODEProblem(func_line!, x0_all, (0, 2*maxR), Mvars, callback=cb_s)
    # sol = solve(prob, Euler(), abstol=1e-4, reltol=1e-3, dt=0.5, dtmin=1e-6, force_dtmin=true)
    sol = nothing
    try
        sol = solve(prob, Euler(), abstol=1e-4, reltol=1e-3, dt=0.5, dtmin=1e-4, force_dtmin=true)
    catch
        sol = solve(prob, Euler(), abstol=1e-4, reltol=1e-3, dt=5e-2, dtmin=1e-4, force_dtmin=true)
    end
    
    
    if length(xc) == 0
        return 0.0, 0.0, 0, 0.0, 0.0, 0.0
    end
    
    if rand_cut
        randInx = rand(1:n_max)
        weights = length(xc)
        indx = 0
        
        
        if weights .>= randInx
            indx = randInx
        else
            return 0.0, 0.0, 0, 0.0, 0.0, 0.0
        end
        
        xpos_flat = [xc[indx] yc[indx] zc[indx]]
        rmag = sqrt.(sum(xpos_flat.^2))

            
        vmag_loc = sqrt.(vIfty_mag.^2 .+ 2 .* GNew .* Mass_NS ./ rmag) ./ c_km
        
        return xpos_flat, rRND, 1, weights, vvec_loc .* vmag_loc[1], vIfty ./ c_km
    else
        xpos_flat = [xc yc zc]
        weights = ones(length(xc))
        rmag = sqrt.(sum(xpos_flat.^2))
        num_c = length(xc)
        
        vmag_loc = sqrt.(vIfty_mag.^2 .+ 2 .* GNew .* Mass_NS ./ rmag) ./ c_km
        
        return xpos_flat, rRND .* ones(num_c), num_c, weights, vvec_loc .* vmag_loc, vIfty ./ c_km .* ones(num_c)
        
    end
    
end



function find_samples(maxR, ntimes_ax, θm, ωPul, B0, rNS, Mass_a, Mass_NS; n_max=8, batchsize=2, thick_surface=false, iso=false, melrose=false, pre_randomized=nothing, flat=false, vmean_ax=220.0)

    ## RE WRITE IN TERMS OF DIFFERENTIAL EQUATION AND CALLBACK!
    tt_ax = LinRange(-1.2*maxR, 1.2*maxR, ntimes_ax); # Not a real physical time -- just used to get trajectory crossing
    cxing = nothing
    
    if isnothing(pre_randomized)
        # ~~~collecting random samples
        
        # randomly sample angles θ, ϕ, hit conv surf
        θi = acos.(1.0 .- 2.0 .* rand(batchsize));
        ϕi = rand(batchsize) .* 2π;
        
        # local velocity
        θi_loc = acos.(1.0 .- 2.0 .* rand(batchsize));
        ϕi_loc = rand(batchsize) .* 2π;
        
        # randomly sample x1 and x2 (rotated vectors in disk perpendicular to (r=1, θ, ϕ) with max radius R)
        ϕRND = rand(batchsize) .* 2π;
        
        # radius on disk
        rRND = sqrt.(rand(batchsize)) .* maxR; # standard flat sampling
        # rRND = rand(batchsize) .* maxR; # 1/r sampling
        
        # ~~~ Done collecting random samples
    else

        θi = acos.(1.0 .- 2.0 .* pre_randomized[:, 1]);
        ϕi = pre_randomized[:, 2] .* 2π;
        
        # local velocity
        θi_loc = acos.(1.0 .- 2.0 .* pre_randomized[:, 3]);
        ϕi_loc = pre_randomized[:, 4] .* 2π;
        
        # randomly sample x1 and x2 (rotated vectors in disk perpendicular to (r=1, θ, ϕ) with max radius R)
        ϕRND = pre_randomized[:, 5] .* 2π;
        
        # radius on disk
        # rRND = sqrt.(pre_randomized[:, 6]) .* maxR; # standard flat sampling
        rRND = pre_randomized[:, 6] .* maxR; # 1/r sampling
    end
    
    # disk direction
    vvec_all = [sin.(θi) .* cos.(ϕi) sin.(θi) .* sin.(ϕi) cos.(θi)];
    
    # vel direction
    vvec_loc = [sin.(θi_loc) .* cos.(ϕi_loc) sin.(θi_loc) .* sin.(ϕi_loc) cos.(θi_loc)];
    
    x1 = rRND .* cos.(ϕRND);
    x2 = rRND .* sin.(ϕRND);
    # rotate using Inv[EurlerMatrix(ϕi, θi, 0)] on vector (x1, x2, 0)
    x0_all= [x1 .* cos.(-ϕi) .* cos.(-θi) .+ x2 .* sin.(-ϕi) x2 .* cos.(-ϕi) .- x1 .* sin.(-ϕi) .* cos.(-θi) x1 .* sin.(-θi)];
    x_axion = [transpose(x0_all[i,:]) .+ transpose(vvec_all[i,:]) .* tt_ax[:] for i in 1:batchsize];
    
    # print(x0_all, "\t", vvec_all, "\n")
    
    vIfty = (vmean_ax .+ rand(batchsize, 3) .* 1.0e-5) ./ sqrt.(3);
    # vIfty = vmean_ax .* erfinv.( 2.0 .* rand(batchsize, 3) .- 1.0);
    
    vIfty_mag = sqrt.(sum(vIfty.^2, dims=2));
    gammaA = 1 ./ sqrt.(1.0 .- (vIfty_mag ./ c_km).^2 )
    erg_inf_ini = Mass_a .* sqrt.(1 .+ (vIfty_mag ./ c_km .* gammaA).^2)
    # Mass_NS = 1.0

    if !thick_surface
        cxing_st = [get_crossings(log.(func_Plasma(x_axion[i], 0.0, θm, ωPul, B0, rNS; sphericalX=false)) .- log.(Mass_a)) for i in 1:batchsize];
        cxing = [apply(cxing_st[i], tt_ax) for i in 1:batchsize];
    else
        
        for i in 1:batchsize
            valF, truth_vals, minV = test_on_shell(x_axion[i], vvec_loc[i,:], vIfty_mag[i], 0.0, θm, ωPul, B0, rNS, Mass_NS, Mass_a; iso=iso, melrose=melrose)
            
            cxing_st = [get_crossings(valF, keep_all=true) for i in 1:batchsize];
            tt_axNew = tt_ax[truth_vals]
            
            if isnothing(cxing)
                cxing = [apply(cxing_st[i], tt_axNew)]
            else
                cxing = [cxing; [apply(cxing_st[i], tt_axNew)]]
            end
        end
        

        
    end
   
    
    randInx = [rand(1:n_max) for i in 1:batchsize];
    
    # see if keep any crossings
    indx_cx = [if length(cxing[i]) .>= randInx[i] i else -1 end for i in 1:batchsize];

    
    # remove those which dont
    randInx = randInx[indx_cx .> 0];
    indx_cx_cut = indx_cx[indx_cx .> 0];

    cxing_short = [cxing[indx_cx_cut][i][randInx[i]] for i in 1:length(indx_cx_cut)];
    weights = [length(cxing[indx_cx_cut][i]) for i in 1:length(indx_cx_cut)];
    
    
    numX = length(cxing_short);
    R_sample = vcat([rRND[indx_cx_cut][i] for i in 1:numX]...);
    erg_inf_ini = vcat([erg_inf_ini[indx_cx_cut][i] for i in 1:numX]...);

    
    vvec_loc = vvec_loc[indx_cx_cut, :];
    vIfty_mag = vcat([vIfty_mag[indx_cx_cut][i] for i in 1:numX]...);
    
    cxing = nothing
    if numX != 0
        
        xpos = [transpose(x0_all[indx_cx_cut[i], :]) .+ transpose(vvec_all[indx_cx_cut[i], :]) .* cxing_short[i] for i in 1:numX];
        vvec_full = [transpose(vvec_all[indx_cx_cut[i],:]) .* ones(1, 3) for i in 1:numX];
        
        
        t_new_arr = LinRange(- abs.(tt_ax[5] - tt_ax[1]), abs.(tt_ax[5] - tt_ax[1]), 1000);
        xpos_proj = [xpos[i] .+ vvec_full[i] .* t_new_arr[:] for i in 1:numX];

        
        if !thick_surface
            cxing_st = [get_crossings(log.(func_Plasma(xpos_proj[i], 0.0, θm, ωPul, B0, rNS; sphericalX=false)) .- log.(Mass_a)) for i in 1:numX];
            cxing = [apply(cxing_st[i], t_new_arr) for i in 1:numX];
        else
            for i in 1:numX
                valF, truth_vals, minV = test_on_shell(xpos_proj[i], vvec_loc[i,:], vIfty_mag[i], 0.0, θm, ωPul, B0, rNS, Mass_NS, Mass_a; iso=iso, melrose=melrose)
                
                cxing_st = [get_crossings(valF, keep_all=true) for i in 1:batchsize];
                tt_axNew = t_new_arr[truth_vals]
                if isnothing(cxing)
                    cxing = [apply(cxing_st[i], tt_axNew)]
                else
                    cxing = [cxing; [apply(cxing_st[i], tt_axNew)]]
                end
            end
        
        # cxing_st = [get_crossings(test_on_shell(xpos_proj[i], vvec_loc[i, :], vIfty_mag[i], 0.0,  θm, ωPul, B0, rNS, Mass_NS, Mass_a; iso=iso, melrose=melrose)) for i in 1:numX];
        end
    
        
        
        indx_cx = [if length(cxing[i]) .> 0 i else -1 end for i in 1:numX];
        indx_cx_cut = indx_cx[indx_cx .> 0];
        R_sample = R_sample[indx_cx_cut];
        erg_inf_ini = erg_inf_ini[indx_cx_cut];
        vvec_loc = vvec_loc[indx_cx_cut, :];
        vIfty_mag = vIfty_mag[indx_cx_cut];
        
        numX = length(indx_cx_cut);
        if numX == 0
            return 0.0, 0.0, 0, 0.0, 0.0, 0.0
        end



        randInx = [rand(1:length(cxing[indx_cx_cut][i])) for i in 1:numX];
        cxing = [cxing[indx_cx_cut][i][randInx[i]] for i in 1:numX];
        vvec_flat = reduce(vcat, vvec_full);
       
        xpos = [xpos[indx_cx_cut[i],:] .+ vvec_full[indx_cx_cut[i],:] .* cxing[i] for i in 1:numX];
        vvec_full = [vvec_full[indx_cx_cut[i],:] for i in 1:numX];

        try
            xpos_flat = reduce(vcat, xpos);
        catch
            print("why is this a rare fail? \t", xpos, "\n")
        end
   
        try
            xpos_flat = reduce(vcat, xpos_flat);
            vvec_flat = reduce(vcat, vvec_full);
        catch
            print("for some reason reduce fail... ", vvec_full, "\t", xpos_flat, "\n")
            vvec_flat = vvec_full;
        end

       
        rmag = sqrt.(sum(xpos_flat .^ 2, dims=2));
        indx_r_cut = rmag .> rNS; #
        
        if sum(indx_r_cut) - length(xpos_flat[:,1 ]) < 0
            xpos_flat = xpos_flat[indx_r_cut[:], :]
            vvec_flat = vvec_flat[indx_r_cut[:], :]
            R_sample = R_sample[indx_r_cut[:]]
            erg_inf_ini = erg_inf_ini[indx_r_cut[:]];
            vvec_loc = vvec_loc[indx_r_cut[:], :];
            vIfty_mag = vIfty_mag[indx_r_cut[:]];
        
            numX = length(xpos_flat);
            rmag = sqrt.(sum(xpos_flat .^ 2, dims=2));
        end
        
        ntrajs = length(R_sample)
        if ntrajs == 0
            return 0.0, 0.0, 0, 0.0, 0.0, 0.0
        end
        
        
        # Renormalize loc velocity and solve asymptotic
        
        vmag_loc = sqrt.(vIfty_mag.^2 .+ 2 .* GNew .* Mass_NS ./ rmag) ./ c_km
       
        v0 = vvec_loc .* vmag_loc
        
        vIfty = zeros(ntrajs, 3)
        ϕ = atan.(xpos_flat[:, 2], xpos_flat[:, 1])
        θ = acos.(xpos_flat[:, 3] ./ rmag)
       
        for i in 1:ntrajs
            for j in 1:3
                vIfty[i, j] = v_infinity(θ[i], ϕ[i], rmag[i], transpose(v0[i, :]); v_comp=j, Mass_NS=Mass_NS);
            end
        end
        
        ωpL = func_Plasma(xpos_flat, zeros(ntrajs), θm, ωPul, B0, rNS; sphericalX=false)
        if ntrajs > 1
            vtot = sqrt.(sum(v0.^2, dims=2))
        else
            vtot = sqrt.(sum(v0.^2))
        end
        gamF = 1 ./ sqrt.(1.0 .- vtot.^2)
        erg_ax = Mass_a .* sqrt.(1.0 .+ (gamF .* vtot).^2) ;
        

      
        # make sure not in forbidden region....
        fails = ωpL .> erg_ax;
        n_fails = sum(fails);
        
        
        
        cxing = nothing;
        if n_fails > 0
            try
                vvec_flat = reduce(vcat, vvec_flat);
            catch
                vvec_flat = vvec_flat;
            end
            print("fails... \n")
      
            if !thick_surface
                ωpLi2 = [if fails[i] == 1 erg_ax .- func_Plasma(transpose(xpos_flat[i,:]) .+ transpose(vvec_flat[i,:]) .* t_new_arr[:], [0.0], θm, ωPul, B0, rNS; sphericalX=false) else -1 end for i in 1:ntrajs];
                ωpLi2 = transpose(reduce(vcat, ωpLi2));
                
                t_new = [if length(ωpLi2[i]) .> 1 t_new_arr[findall(x->x==ωpLi2[i][ωpLi2[i] .> 0][argmin(ωpLi2[i][ωpLi2[i] .> 0])], ωpLi2[i])][1] else -1e6 end for i in 1:length(ωpLi2)];
                t_new = t_new[t_new .> -1e6];
                xpos_flat[fails[:],:] .+= vvec_flat[fails[:], :] .* t_new;
                
            else
                xpos_proj = [transpose(xpos_flat[i,:]) .+ transpose(vvec_flat[i,:]) .* t_new_arr[:] for i in 1:ntrajs];
                ## FIXING
                for i in 1:numX
                    valF, truth_vals, minV = test_on_shell(xpos_proj[i], vvec_loc[i,:], vIfty_mag[i], 0.0, θm, ωPul, B0, rNS, Mass_NS, Mass_a; iso=iso, melrose=melrose)
                    cxing_st = [get_crossings(valF, keep_all=true) for i in 1:batchsize];
                    tt_axNew = t_new_arr[truth_vals]
                    if isnothing(cxing)
                        cxing = [apply(cxing_st[i], tt_axNew)]
                    else
                        cxing = [cxing; [apply(cxing_st[i], tt_axNew)]]
                    end
                end
                
                xpos_flat = [xpos_flat[i,:] .+ vvec_flat[i,:] .* cxing[i] for i in 1:ntrajs];
                try
                    xpos_flat = reduce(vcat, xpos_flat);
                catch
                    xpos_flat = xpos_flat;
                end
            end
            
        end
        
        return xpos_flat, R_sample, ntrajs, weights, v0, vIfty
    else
        return 0.0, 0.0, 0, 0.0, 0.0, 0.0
    end
    
end

end



function write_to_file(fname, sve_array, Mass_a, Ax_g, θm, ωPul, B0, rNS, Mass_NS, flat, isotropic, melrose, add_tau, fix_time, delta_on_v, rho_DM, v_NS, vmean_ax, null_fill, dead, dead_rmax; sve_mode=1)
    fid = h5open(fname, "w")
    

    # save generic info
    fid["Mass_Ax"] = [Mass_a];
    fid["Ax_g"] = [Ax_g];
    fid["MisAngle"] = [θm];
    fid["RotFreq"] = [ωPul];
    fid["Period"] = [2*pi ./ ωPul];
    fid["B0"] = [B0];
    fid["rNS"] = [rNS];
    fid["Mass_NS"] = [Mass_NS];
    fid["v_NS"] = v_NS;
    fid["rhoDM"] = [rho_DM];
    fid["vmean_ax"] = [vmean_ax];
    fid["eta_fill"] = [null_fill];
    
    
    if flat
        fid["is_flat"] = [1];
    else
        fid["is_flat"] = [0];
    end
    if isotropic
        fid["is_isotropic"] = [1];
    else
        fid["is_isotropic"] = [0];
    end
    if melrose
        fid["use_melrose"] = [1];
    else
        fid["use_melrose"] = [0];
    end
    if add_tau
        fid["add_tau"] = [1];
    else
        fid["add_tau"] = [0];
    end
    fid["fix_time"] = [fix_time];
    if delta_on_v
        fid["delta_on_v"] = [1];
    else
        fid["delta_on_v"] = [0];
    end
    if dead
        fid["dead"] = [1];
    else
        fid["dead"] = [0];
    end
    fid["dead_rmax"] = [dead_rmax];
    
    # now the main stuff
    fid["thetaK_final"] = sve_array[:, 1]
    fid["phiK_final"] = sve_array[:, 2]
    fid["thetaX_final"] = sve_array[:, 3]
    fid["phiX_final"] = sve_array[:, 4]
    fid["weights"] = sve_array[:, 5]
    fid["erg_final"] = sve_array[:, 6]

    
    if sve_mode == 1
        fid["r_init"] = sve_array[:, 7]
        fid["theta_init"] = sve_array[:, 8]
        fid["phi_init"] = sve_array[:, 9]
        fid["kr_init"] = sve_array[:, 10]
        fid["ktheta_init"] = sve_array[:, 11]
        fid["kphi_init"] = sve_array[:, 12]
        fid["red_factor"] = sve_array[:, 13]
        fid["erg_loc"] = sve_array[:, 14]
        fid["plasmaF"] = sve_array[:, 15]
        fid["Bmag"] = sve_array[:, 16]
        fid["kmag"] = sve_array[:, 17]
        fid["dense_extra"] = sve_array[:, 18]
        fid["Prob"] = sve_array[:, 19]
        # fid["Lc_Calc"] = sve_array[:, 20]
        fid["Lc_deriv"] = sve_array[:, 21]
        fid["cos_k_gradE"] = sve_array[:, 22]
        fid["g_det"] = sve_array[:, 23]
        fid["theta_kB"] = sve_array[:, 24]
        fid["vGroup"] = sve_array[:, 25]
        fid["grad_Emag"] = sve_array[:, 26]
        fid["tau"] = sve_array[:, 27]
        fid["cos_k_gradE_2term"] = sve_array[:, 28]
        fid["grad_Emag_2term"] = sve_array[:, 29]
        fid["cos_k_gradE_3term"] = sve_array[:, 30]
    end
    
    close(fid)
    return
end

function main_runner(Mass_a, Ax_g, θm, ωPul, B0, rNS, Mass_NS, Ntajs; flat=true, isotropic=false, melrose=false, CP_one_D=false, ode_err=1e-5, fix_time=0.0, add_tau=false, file_tag="", ntimes=4, v_NS=[0 0 0], rho_DM=0.45, save_more=false, vmean_ax=220.0, dir_tag="results", n_maxSample=6, thick_surface=false, iseed=Nothing, sve_mode=1, plasma_mod="GJ", mag_mod="Dipole", lambdaM=1.0, psi=0.0, delta_on_v=true, θmQ=0.0, phiQ=0.0, B_DQ=0.0, ray_trace=true, reflect_LFL=false, null_fill=0.0, dead=false, dead_rmax=30.0)

    
    if iseed != Nothing
        Random.seed!(iseed)
    end
    
    batchsize = 1 # hard coded! cannot be changed due to ridiculous reasons. also no reason to change.
    gammaF = [1.0, 1.0] # need to delete this, what a stupid idea.
    
    ##### define plasma and mag model
    global func_BField = define_B_model(mag_mod,  θmQ=θmQ, phiQ=phiQ, B_DQ=B_DQ, Mass_NS=Mass_NS)
    rho_twist = false
    if mag_mod == "Magnetar"
        rho_twist = true
    end
    global func_Plasma = define_plasma_model(plasma_mod, func_BField, lambdaM=lambdaM, psi=psi, Mass_NS=Mass_NS, rho_twist=rho_twist, reflect_LFL=reflect_LFL, null_fill=null_fill, Mass_a=Mass_a, dead=dead, dead_rmax=dead_rmax)
    RT = RayTracerGR; # define ray tracer module
    RT.magneto_define(Mass_a, plasma_mod, mag_mod, lambdaM, psi, θmQ=θmQ, phiQ=phiQ, B_DQ=B_DQ, Mass_NS=Mass_NS, reflect_LFL=reflect_LFL, null_fill=null_fill, dead=dead, dead_rmax=dead_rmax);

    #### check we can actually generate samples
    maxR = RT.Find_Conversion_Surface(Mass_a, fix_time, θm, ωPul, B0, rNS, 1, false, plasma_mod, func_BField, dead=dead, dead_rmax=dead_rmax)[1]
    print("Max R In Surface Finder \t", maxR, "\n")
    if maxR .< rNS
        print("Too small Max R.... quitting.... \n")
        omegaP_test = RT.func_Plasma(rNS .* [sin.(θm) 0.0 cos.(θm)], 0.0, θm, ωPul, B0, rNS; sphericalX=false);
        print("Max omegaP found... \t", omegaP_test, "Max radius found...\t", maxR, "\n")
        fileN = File_Name_Out(Mass_a, Ax_g, Mass_NS, rNS, v_NS, B0, 2 .* pi ./ ωPul, θm, null_fill, Ntajs; file_tag=file_tag, mag_mod=mag_mod, B_DQ=B_DQ, θQ=θmQ, reflect=reflect_LFL, dead=dead, dead_rmax=dead_rmax)
        
        saveAll = [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.0 0 0 0 0 0 0 [0]' 0 0 0]
        write_to_file(fileN, saveAll, Mass_a, Ax_g, θm, ωPul, B0, rNS, Mass_NS, flat, isotropic, melrose, add_tau, fix_time, delta_on_v, rho_DM, v_NS, vmean_ax, null_fill, dead, dead_rmax; sve_mode=sve_mode)
        return
    end
    #####

    photon_trajs = 1
    desired_trajs = Ntajs
    # assumes desired_trajs large!
    
    # SaveAll = zeros(desired_trajs * 2, 31);
    saveAll = nothing;
    f_inx = 0;

    
    ln_t_start = -20;
    ln_t_end = log.(10.0 ./ ωPul);

    NumerPass = [ln_t_start, ln_t_end, ode_err];
    ttΔω = exp.(LinRange(ln_t_start, ln_t_end, ntimes));

    
    if (v_NS[1] == 0)&&(v_NS[1] == 0)&&(v_NS[1] == 0)
        phaseApprox = true;
    else
        phaseApprox = false;
    end
    vNS_mag = sqrt.(sum(v_NS.^2));
    if vNS_mag .> 0
        vNS_theta = acos.(v_NS[3] ./ vNS_mag);
        vNS_phi = atan.(v_NS[2], v_NS[1]);
    end
    
    # define time at which we find crossings
    t0_ax = zeros(batchsize);
    xpos_flat = zeros(batchsize, 3);
    velNorm_flat = zeros(batchsize, 3);
    vIfty = zeros(batchsize, 3);
    R_sample = zeros(batchsize);
    mcmc_weights = zeros(batchsize);
    filled_positions = false;
    fill_indx = 1;
    fail_count = 0;

    small_batch = 1
    
    # Random.seed!(iseed)
    # rndX = 0
    # TotalTime = nothing
    
    
    while photon_trajs < desired_trajs
    
        while !filled_positions
            # Random.seed!(iseed + rndX)
            # rndX += 1
            

            xv, Rv, numV, weights, vvec_in, vIfty_in = RT.find_samples_new(maxR, θm, ωPul, B0, rNS, Mass_a, Mass_NS; n_max=n_maxSample, batchsize=small_batch, thick_surface=thick_surface, iso=isotropic, melrose=false, flat=false, delta_on_v=delta_on_v)
                     
            f_inx += small_batch
            
            if numV == 0
                continue
            end
          

            for i in 1:Int(numV) # Keep more?
                f_inx -= 1
                if fill_indx <= batchsize
            
                    xpos_flat[fill_indx, :] .= xv[i, :];
                    R_sample[fill_indx] = Rv[i];
                    mcmc_weights[fill_indx] = n_maxSample;
                    velNorm_flat[fill_indx, :] .= vvec_in[i, :];
                    vIfty[fill_indx, :] .= vIfty_in[i, :];
                    fill_indx += 1
                    
                end
            end
            
            if fill_indx > batchsize
                filled_positions = true
                fill_indx = 1
            end
            
            if (f_inx > 5000)&&(photon_trajs == 1)
                print("Failing to find any points, exiting... \n")
                return
            end
        end
        
        
        
        filled_positions = false;
    
        
        rmag = sqrt.(sum(xpos_flat.^ 2, dims=2));
        vIfty_mag = sqrt.(sum(vIfty.^2, dims=2));
        vel_eng = sum((vIfty).^ 2, dims = 2) ./ 2;
        gammaA = 1 ./ sqrt.(1.0 .- (vIfty_mag ).^2 )
        erg_inf_ini = Mass_a .* sqrt.(1 .+ (vIfty_mag  .* gammaA).^2)
        vmag = sqrt.(2 * GNew .* Mass_NS ./ rmag) ; # km/s
        vmag_tot = sqrt.(vmag .^ 2 .+ vIfty_mag.^2); # km/s
        
        theta_sf = acos.(xpos_flat[:,3] ./ rmag)
        x0_pl = [rmag theta_sf atan.(xpos_flat[:,2], xpos_flat[:,1])]
        g_tt, g_rr, g_thth, g_pp = RT.g_schwartz(x0_pl, Mass_NS);
        erg_ax = erg_inf_ini ./ sqrt.(1.0 .- 2 * GNew .* Mass_NS ./ rmag ./ c_km.^2 );
        
        
        k_init = RT.k_norm_Cart(xpos_flat, velNorm_flat, 0.0, erg_inf_ini, θm, ωPul, B0, rNS, Mass_NS, Mass_a, melrose=melrose, flat=false, isotropic=isotropic)
        v_tot = (vIfty_mag .+ vmag) ./ c_km
        gamma_tot = 1 ./ sqrt.(1.0 .- v_tot.^2 )

        
        MagnetoVars = [θm, ωPul, B0, rNS, gammaF, zeros(batchsize), Mass_NS, Mass_a, erg_ax, erg_inf_ini, flat, isotropic, melrose]
        

        
        if ray_trace
            if null_fill > 0
                dt_minCut = 1.0e-9
            else
                dt_minCut = 1.0e-7
            end
            xF, kF, tF, fail_indx, optical_depth = RT.propagate(xpos_flat, k_init, ntimes, MagnetoVars, NumerPass, dt_minCut=dt_minCut);
            fail_count += sum(fail_indx .== 0.0)
        else
            xF = ones(1,3,4)
            kF = ones(1,3,4)
            tF = [0.0]
            fail_indx = [1.0]
            optical_depth = 0.0
        end
        
        

        jacobian_GR = RT.g_det(x0_pl, zeros(batchsize), θm, ωPul, B0, rNS, Mass_NS; flat=false); # unitless
        
        jac_fv = zeros(batchsize)
        for i in 1:batchsize
            jac_fv[i] = (vmag_tot[i] ./ vIfty_mag[i]) ./ c_km
            # velVT, accur = RT.solve_vel_CS(x0_pl[i, 2], x0_pl[i, 3], x0_pl[i, 1], vIfty[i, :], errV=1e-10)
            # jacT = jacobian(x ->RT.v_infinity(x0_pl[i, 2], x0_pl[i, 3], x0_pl[i, 1], [x[1] x[2] x[3]],  v_comp=4, Mass_NS=1), transpose(velV))
        end
        
        
        
        
        
        
        # Get local properties needed for everything!
        ωp = func_Plasma(x0_pl, fix_time, θm, ωPul, B0, rNS, zeroIn=false);
        Bsphere = func_BField(xpos_flat, zeros(batchsize), θm, ωPul, B0, rNS; Mass_NS=Mass_NS, flat=false)
        
        ksphere = RT.k_sphere(xpos_flat, k_init, θm, ωPul, B0, rNS, zeros(batchsize), Mass_NS, false)
        Bmag = sqrt.(RT.spatial_dot(Bsphere, Bsphere, batchsize, x0_pl, Mass_NS)) .* 1.95e-2; # eV^2
        kmag = sqrt.(RT.spatial_dot(ksphere, ksphere, batchsize, x0_pl, Mass_NS));
        ctheta_B = RT.spatial_dot(Bsphere, ksphere, batchsize, x0_pl, Mass_NS) .* 1.95e-2 ./ (kmag .* Bmag)
        stheta_B = sin.(acos.(ctheta_B))
        if isotropic
            ctheta_B .*= 0.0
            stheta_B ./= stheta_B
        end
        
        v_group = RT.grad(RT.omega_function(x0_pl, RT.seed(ksphere), 0.0, -erg_inf_ini, θm, ωPul, B0, rNS, Mass_NS, flat=false, iso=isotropic, melrose=true)) #
        v_group[:, 1] ./= g_rr
        v_group[:, 2] ./= g_thth
        v_group[:, 3] ./= g_pp
        vgNorm = sqrt.(RT.spatial_dot(v_group, v_group, batchsize, x0_pl, Mass_NS));
        
    
        
        
        Mvars =  [θm, ωPul, B0, rNS, gammaF, zeros(batchsize), Mass_NS, Mass_a, false, isotropic, erg_ax]
        local_vars = [ωp, Bsphere, Bmag, ksphere, kmag, vgNorm, ctheta_B, stheta_B]
        Prob, sln_δE, cos_w, grad_Emag, cos_w_2, grad_Emag_2, cos_w_3 = RT.conversion_prob(Ax_g, x0_pl, Mvars, local_vars, one_D=CP_one_D)
        
        
        
        
        
        # RT.dEgamma_ds(x0_pl, Mvars, local_vars)
        conv_length_E = sqrt.(pi ./ (sln_δE ./ hbar ./ c_km)) # km
        

        redshift_factor = sqrt.(1.0 .- 2 * GNew .* Mass_NS ./ rmag ./ c_km.^2 );
        dense_extra = 2 ./ sqrt.(π) * (1.0 ./ (vmean_ax ./ c_km)) .* sqrt.(2.0 * Mass_NS * GNew / c_km^2 ./ x0_pl[:,1])
        local_n = (rho_DM .* 1e9 ./ Mass_a) .* dense_extra[:]
        
        # factor of 1/2 from cos angle sampling
        phaseS = (2 .* π  .* 2 .* maxR.^2 ./ 2) .* Prob .*  local_n # [km^2 * eV] .last factor is number density. Notice first part: 2*pi [phi], R^2/2 [r], 2 [alpha]


        
        
        # conversion_F = ones(length(vmag_tot))
        # f_infty = (vIfty_mag ./ c_km).^2 ./ (π .* (vmean_ax ./ c_km).^2).^(3.0 ./ 2) .* exp.(- (vIfty_mag ./ vmean_ax).^2);
        # f_infty = 1.0 ./ (4 .* π);
        f_infty = 1.0
        # sln_prob = v_tot .* abs.(cos.(angleVal)) .* phaseS .* (1e5 .^ 2) .* c_km .* 1e5 .* mcmc_weights .*  (jac_fv .* f_infty) .* fail_indx .* jacobian_GR[:]; # photons / second ONLY USE FOR NON DELTA!
        
        
        # CURRENT ONE
        sln_prob = abs.(cos_w[:]) .* kmag[:] .* phaseS .* (1e5 .^ 2) .* c_km .* 1e5 .* mcmc_weights .* fail_indx .* jacobian_GR[:]; # photons / second
        
        ϕf = atan.(kF[:, 2, end], kF[:, 1, end]);
        ϕfX = atan.(xF[:, 2, end], xF[:, 1, end]);
        θf = acos.(kF[:, 3, end] ./ sqrt.(sum(kF[:, :, end] .^2, dims=2)));
        θfX = acos.(xF[:, 3, end] ./ sqrt.(sum(xF[:, :, end] .^2, dims=2)));
        
        # compute energy dispersion ωf
        Δω = tF[:, end] .* redshift_factor;
        
        if add_tau
            sln_prob[:] .* exp.(-optical_depth)
        end
        
        num_photons = length(ϕf)
        
        f_inx += num_photons
        finalWeight = sln_prob[:];
        
        if sve_mode == 0
            row = [θf ϕf θfX ϕfX finalWeight Δω]
        elseif sve_mode == 1
            row = [θf ϕf θfX ϕfX finalWeight Δω x0_pl[:, 1] x0_pl[: ,2] x0_pl[:, 3] ksphere[:, 1] ksphere[: ,2] ksphere[:, 3] redshift_factor erg_ax ωp Bmag kmag dense_extra Prob 0.0 conv_length_E cos_w jacobian_GR acos.(ctheta_B[:]) vgNorm grad_Emag [optical_depth]' cos_w_2 grad_Emag_2 cos_w_3]
        end
        
        if isnothing(saveAll)
            saveAll = row
        else
            saveAll = [saveAll; row]
        end
       
        photon_trajs += num_photons;
        

        if mod(photon_trajs, 100) == 0
            GC.gc();
        end
        
        
    end
    print("fail count \t", fail_count, "\n")
    # cut out unused elements
    saveAll = saveAll[saveAll[:, 5] .> 0, :];
    saveAll[:, 5] .*= 1 ./ ( float(f_inx) ); # divide off by N trajectories sampled
    

    fileN = File_Name_Out(Mass_a, Ax_g, Mass_NS, rNS, v_NS, B0, 2 .* pi ./ ωPul, θm, null_fill, Ntajs; file_tag=file_tag, mag_mod=mag_mod, B_DQ=B_DQ, θQ=θmQ, reflect=reflect_LFL, dead=dead, dead_rmax=dead_rmax)
  
    write_to_file(fileN, saveAll, Mass_a, Ax_g, θm, ωPul, B0, rNS, Mass_NS, flat, isotropic, melrose, add_tau, fix_time, delta_on_v, rho_DM, v_NS, vmean_ax, null_fill, dead, dead_rmax; sve_mode=sve_mode)


end

