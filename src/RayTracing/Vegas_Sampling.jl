# __precompile__()

include("RayTracer_Fast_GR_2.jl")
# using MonteCarloIntegration
include("Constants.jl")
include("Vegas_Algorithim.jl")
include("environment.jl")
using Distributions
using Statistics

RT = RayTracerGR; # define ray tracer module


function Vegas_sampler(theta_target, theta_err, Mass_a, θm, ωPul, B0, rNS; return_width=false, maxiter=10, debug=false, nbins=4, ncalls=1000, constrain_phi=false, phi_target=0.0, phi_err=0.05, fix_time=0.0, Mass_NS=1.0, thick_surface=false, flat=true, isotropic=false, melrose=false, ode_err=1e-7, rho_DM=0.45, vmean_ax=220.0, Ax_g=1e-12, add_tau=false, CLen_Scale=false, CP_one_D=false, plasma_mod="GJ", mag_mod="Dipole", lambdaM=1.0, psi=0.0, delta_on_v=true, θmQ=0.0, phiQ=0.0, B_DQ=0.0, ray_trace=true, reflect_LFL=false, null_fill=0.0, dead=false, dead_rmax=30.0, Ntest_ini=1000)

    rho_twist = false
    if mag_mod == "Magnetar"
        rho_twist = true
    end
    
    global func_BField = define_B_model(mag_mod, θmQ=θmQ, phiQ=phiQ, B_DQ=B_DQ, Mass_NS=Mass_NS)
    global func_Plasma = define_plasma_model(plasma_mod, func_BField, lambdaM=lambdaM, psi=psi, Mass_NS=Mass_NS, rho_twist=rho_twist, reflect_LFL=reflect_LFL, null_fill=null_fill, Mass_a=Mass_a, dead=dead, dead_rmax=dead_rmax)
    RT = RayTracerGR; # define ray tracer module
    RT.magneto_define(Mass_a, plasma_mod, mag_mod, lambdaM, psi, θmQ=θmQ, phiQ=phiQ, B_DQ=B_DQ, Mass_NS=Mass_NS, reflect_LFL=reflect_LFL, null_fill=null_fill, dead=dead, dead_rmax=dead_rmax);

    
    
    maxR = RT.Find_Conversion_Surface(Mass_a, fix_time, θm, ωPul, B0, rNS, 1, false, plasma_mod, func_BField, dead=dead, dead_rmax=dead_rmax)[1]
    print(maxR, "\n")
    st = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    en = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    f = x -> vegas_func(x, maxR, Mass_a, θm, ωPul, B0, rNS; fix_time=fix_time, Mass_NS=Mass_NS, thick_surface=thick_surface, flat=flat, isotropic=isotropic, melrose=melrose, rho_DM=rho_DM, Ax_g=Ax_g, theta_mean=theta_target, theta_err=theta_err, phi_mean=phi_target, phi_err=phi_err, constrain_phi=constrain_phi, add_tau=add_tau, CLen_Scale=CLen_Scale, CP_one_D=CP_one_D, finalX_test=true, vmean_ax=vmean_ax);
    
    
    finalDist = nothing
    for i in 1:Ntest_ini
        theta_out = f(rand(6))
        
        if theta_out .> 1e-100
            if isnothing(finalDist)
                finalDist = [abs.(theta_target .- theta_out)]
            else
                finalDist = [finalDist; abs.(theta_target .- theta_out)]
            end
        end
    end
    # print("HERE \t", finalDist, "\n")
    theta_test = quantile!(finalDist, 0.01)
    if theta_test .>= theta_err
        # this is max err threshold...
        if theta_test .> 0.05
            theta_err = 0.05
            ncalls *= 2
        elseif theta_test .< 0.02
            theta_err = 0.02
        else
            theta_err = theta_test
        end
    end
    print("Using Theta Bin Size \t", theta_err, "\n\n")

    f = x -> vegas_func(x, maxR, Mass_a, θm, ωPul, B0, rNS; fix_time=fix_time, Mass_NS=Mass_NS, thick_surface=thick_surface, flat=flat, isotropic=isotropic, melrose=melrose, rho_DM=rho_DM, Ax_g=Ax_g, theta_mean=theta_target, theta_err=theta_err, phi_mean=phi_target, phi_err=phi_err, constrain_phi=constrain_phi, add_tau=add_tau, CLen_Scale=CLen_Scale, CP_one_D=CP_one_D, finalX_test=false, delta_on_v=delta_on_v, vmean_ax=vmean_ax);
    
    vegas_result = vegas(f, st, en, nbins = nbins, ncalls = ncalls, maxiter=maxiter, debug=debug)

    print(vegas_result.integral_estimate, "\n")
    print(vegas_result.standard_deviation, "\n")
    print(vegas_result.chi_squared_average, "\n")
    
    
    if return_width
        output_W = zeros(ncalls);
        output_E = zeros(ncalls);
        # new_X = sample_from_adaptive_grid(vegas_result, ncalls)
        ndim = length(st)
        ymat = QuasiMonteCarlo.sample(ncalls, zeros(ndim), ones(ndim), QuasiMonteCarlo.UniformSample())
        
        from_y_to_i(y) = floor(Int,nbins*y) + 1
		delta(y) = y*nbins + 1 - from_y_to_i(y)
        x = vegas_result.adaptive_grid
        delx = vegas_result.grid_spacing
		from_y_to_x(y,dim) = x[from_y_to_i(y),dim] + delx[from_y_to_i(y),dim]*delta(y)
		J(y,dim) = nbins*delx[from_y_to_i(y),dim]
        xmat = similar(ymat)
		for dim = 1:ndim
			xmat[dim,:] .= map(y -> from_y_to_x(y, dim), ymat[dim,:])
		end
		Js = zeros(eltype(ymat), size(ymat, 2))
		for i = 1:size(ymat,2)
			Js[i] = prod(J(ymat[dim, i],dim) for dim = 1:ndim)
		end
        
        
        for i in 1:ncalls
            output_W_hold, output_E_hold = vegas_func(transpose(xmat)[i, :], maxR, Mass_a, θm, ωPul, B0, rNS; fix_time=fix_time, Mass_NS=Mass_NS, thick_surface=thick_surface, flat=flat, isotropic=isotropic, melrose=melrose, rho_DM=rho_DM, Ax_g=Ax_g, theta_mean=theta_target, theta_err=theta_err, phi_mean=phi_target, phi_err=phi_err, constrain_phi=constrain_phi, add_tau=add_tau, CLen_Scale=CLen_Scale, CP_one_D=CP_one_D, finalX_test=false, return_width=true, delta_on_v=delta_on_v, vmean_ax=vmean_ax);
            
            output_W[i] = output_W_hold;
            output_E[i] = output_E_hold;
        end
        
        
        meanE = sum(output_W .* Js .* output_E) ./ sum(output_W .* Js)
        std_E = sqrt.(sum(output_W .* Js .* (output_E .- meanE).^2) ./ sum(meanE .^2 .* output_W .* Js))
        # print("Final test \t", sum(output_W .* Js), "\t", vegas_result.integral_estimate, "\t", meanE, "\t", std_E, "\n")
        return vegas_result.integral_estimate, vegas_result.standard_deviation, vegas_result.chi_squared_average, meanE, std_E
    else
        return vegas_result.integral_estimate, vegas_result.standard_deviation, vegas_result.chi_squared_average, Mass_a, 1.0
    end
end


function vegas_func(input_samples, maxR, Mass_a, θm, ωPul, B0, rNS; fix_time=0.0, Mass_NS=1.0, thick_surface=false, flat=true, isotropic=false, melrose=false, ode_err=1e-5, theta_mean=1.0, phi_mean=0.0, theta_err=0.05, phi_err=0.05, rho_DM=0.45, vmean_ax=220.0, Ax_g=1e-12, constrain_phi=false, n_maxSample=6, add_tau=false, CLen_Scale=false, CP_one_D=CP_one_D, finalX_test=false, return_width=false, delta_on_v=false)
    
    
    xpos_flat, Rv, numV, weights, velNorm_flat, vIfty = RT.find_samples_new(maxR, θm, ωPul, B0, rNS, Mass_a, Mass_NS; thick_surface=thick_surface, iso=isotropic, melrose=melrose, pre_randomized=input_samples, flat=flat, rand_cut=true, batchsize=1, n_max=n_maxSample, delta_on_v=delta_on_v, vmean_ax=vmean_ax)
    
    
    
    if numV == 0
        if !return_width
            return 1e-100
        else
            return 1e-100, 1.0
        end
    end
    
    

    ln_t_start = -20; #
    ln_t_end = log.(1 ./ ωPul);
    NumerPass = [ln_t_start, ln_t_end, ode_err];
    ntimes = 3
    
    
    vIfty_mag = sqrt.(sum(vIfty.^2, dims=2));
    
    gammaA = 1 ./ sqrt.(1.0 .- (vIfty_mag ).^2 )
    erg_inf_ini = Mass_a .* sqrt.(1 .+ (vIfty_mag  .* gammaA).^2)
    k_init = RT.k_norm_Cart(xpos_flat, velNorm_flat, 0.0, erg_inf_ini, θm, ωPul, B0, rNS, Mass_NS, Mass_a, melrose=melrose, flat=flat, isotropic=isotropic, ax_fix=true)
    
    rmag = sqrt.(sum(xpos_flat.^ 2, dims=2));
    theta_sf = acos.(xpos_flat[:,3] ./ rmag)
    x0_pl = [rmag theta_sf atan.(xpos_flat[:,2], xpos_flat[:,1])]
    
    erg_ax = erg_inf_ini ./ sqrt.(1.0 .- 2 * GNew .* Mass_NS ./ rmag ./ c_km.^2 );
    
    MagnetoVars = [θm, ωPul, B0, rNS, [1.0, 1.0], zeros(numV), Mass_NS, Mass_a, erg_ax, erg_inf_ini, flat, isotropic, melrose]
    
    xF, kF, tF, fail_indx, optical_depth = RT.propagate(xpos_flat, k_init, ntimes, MagnetoVars, NumerPass);
    
    
    
    ϕfX = atan.(view(xF, :, 2, ntimes), view(xF, :, 1, ntimes));
    θfX = acos.(view(xF, :, 3, ntimes) ./ sqrt.(sum(view(xF, :, :, ntimes) .^2, dims=2)));
    
    if finalX_test
        return θfX[1]
    end
    
    jacobian_GR = RT.g_det(x0_pl, zeros(numV), θm, ωPul, B0, rNS, Mass_NS; flat=flat); # unitless
    
    ωp = func_Plasma(x0_pl, fix_time, θm, ωPul, B0, rNS, zeroIn=false);
    
    
    Bsphere = func_BField(xpos_flat, zeros(numV), θm, ωPul, B0, rNS; Mass_NS=Mass_NS, flat=flat)
    ksphere = RT.k_sphere(xpos_flat, k_init, θm, ωPul, B0, rNS, zeros(numV), Mass_NS, flat)
    Bmag = sqrt.(RT.spatial_dot(Bsphere, Bsphere, numV, x0_pl, Mass_NS)) .* 1.95e-2; # eV^2
    kmag = sqrt.(RT.spatial_dot(ksphere, ksphere, numV, x0_pl, Mass_NS));
    ctheta_B = RT.spatial_dot(Bsphere, ksphere, numV, x0_pl, Mass_NS) .* 1.95e-2 ./ (kmag .* Bmag)
    stheta_B = sin.(acos.(ctheta_B))
    if isotropic
        ctheta_B .*= 0.0
        stheta_B ./= stheta_B
    end
    g_tt, g_rr, g_thth, g_pp = RT.g_schwartz(x0_pl, Mass_NS);
    v_group = RT.grad(RT.omega_function(x0_pl, RT.seed(ksphere), 0.0, -erg_inf_ini, θm, ωPul, B0, rNS, Mass_NS, flat=flat, iso=isotropic, melrose=true)) #
    v_group[:, 1] ./= g_rr
    v_group[:, 2] ./= g_thth
    v_group[:, 3] ./= g_pp
    vgNorm = sqrt.(RT.spatial_dot(v_group, v_group, 1, x0_pl, Mass_NS));
    
    Mvars =  [θm, ωPul, B0, rNS, [1.0 1.0], zeros(1), Mass_NS, Mass_a, false, isotropic, erg_ax]
    local_vars = [ωp, Bsphere, Bmag, ksphere, kmag, vgNorm, ctheta_B, stheta_B]
    Prob, sln_δE, cos_w, grad_Emag, cos_w_2, grad_Emag_2, cos_w_3 = RT.conversion_prob(Ax_g, x0_pl, Mvars, local_vars, one_D=CP_one_D)
    
    conv_length_E = sqrt.(pi ./ (sln_δE ./ hbar ./ c_km)) # km


    mcmc_weights = n_maxSample
    redshift_factor = sqrt.(1.0 .- 2 * GNew .* Mass_NS ./ rmag ./ c_km.^2 );
    dense_extra = 2 ./ sqrt.(π) * (1.0 ./ (vmean_ax ./ c_km)) .* sqrt.(2.0 * Mass_NS * GNew / c_km^2 ./ x0_pl[:,1])
    local_n = (rho_DM .* 1e9 ./ Mass_a) .* dense_extra[:]
        

    phaseS = (2 .* π  .* 2 .* maxR.^2 ./ 2) .* Prob .* local_n # [km^2 * eV]
    
    sln_prob = abs.(cos_w[:]) .* kmag[:] .* phaseS .* (1e5 .^ 2) .* c_km .* 1e5 .* mcmc_weights .* fail_indx .* jacobian_GR[:]; # photons / second
    
    if add_tau
        sln_prob[:] .* exp.(-optical_depth)
    end
    
    
    finalV = sln_prob .* exp.(-(θfX .- theta_mean).^2 ./ (2 .* theta_err.^2)) ./ sqrt.(2 .* pi .* theta_err.^2) ./ abs.(sin.(θfX))
    
    if constrain_phi
        finalV .+= exp.(-(ϕfX .- phi_mean).^2 ./ (2 .* phi_err.^2)) ./ sqrt.(2 .* pi .* phi_err.^2)
    else
        finalV ./= (2 .* pi ./ ωPul) # period averaged value
    end
    
    
    if !return_width
        return sum(finalV)
    else
        Δω = tF[end];
        return sum(finalV), Δω
    end
end
