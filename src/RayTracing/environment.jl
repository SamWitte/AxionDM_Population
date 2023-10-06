include("Constants.jl")

function define_plasma_model(type, funcB; lambdaM=1.0, psi=0.2, null_fill=0.0, Mass_a=1e-5, Mass_NS=1.0, flat=false, rho_twist=false, reflect_LFL=false, reflect_norm=1e-2, B_DQ=0.0, dead=false, dead_rmax=30.0)
    if type == "GJ"
        
        function GJ_eval(x, t, θm, ω, B0, rNS; zeroIn=true, sphericalX=true, time_deriv=false)
            return GJ_plasma(x, t, θm, ω, B0, rNS, funcB; zeroIn=zeroIn, sphericalX=sphericalX, null_fill=null_fill, Mass_a=Mass_a, Mass_NS=Mass_NS, rho_twist=rho_twist, reflect_LFL=reflect_LFL, reflect_norm=reflect_norm, time_deriv=time_deriv, B_DQ=B_DQ, dead=dead, dead_rmax=dead_rmax)
        end
       
    elseif type == "GJ_Conservative"
        return ellipse_plasma
        
    elseif type == "SphereTest"
        return Spherical_Plasma_Test
    
    elseif type == "Magnetar"
        
        function Magnetar_eval(x, t, θm, ω, B0, rNS; zeroIn=true, sphericalX=true)
            return Magnetar_plasma(x, t, θm, ω, B0, rNS; lambdaM=lambdaM, psi=psi, zeroIn=zeroIn, sphericalX=sphericalX)
        end
        
    end
    
    
end

function define_B_model(type; θmQ=0.0, phiQ=0.0, B_DQ=0.0, Mass_NS=1.0)
    if type == "Dipole"
        return B_dipolar
    elseif type == "DiQuad"
        function GJ_eval(x, t, θm, ω, B0, rNS; Mass_NS=1.0, flat=false, sphericalX=false, return_comp=-1, include_metric=true, sphere_out=true)
            return B_Di_Quad(x, t, θm, θmQ, phiQ, ω, B0, B_DQ, rNS; Mass_NS=Mass_NS, flat=false, sphericalX=sphericalX, return_comp=return_comp, include_metric=include_metric, sphere_out=sphere_out)
        end
    end
end

function GJ_plasma(x, t, θm, ω, B0, rNS, funcB; zeroIn=true, sphericalX=true, null_fill=0.0, Mass_a=1e-5, Mass_NS=1.0, flat=false, rho_twist=false, reflect_LFL=false, reflect_norm=1e-2, time_deriv=false, B_DQ=0.0, dead=false, dead_rmax=30.0)
    # For GJ model, return \omega_p [eV]
    # Assume \vec{x} is in Cartesian coordinates [km], origin at NS, z axis aligned with ω
    # theta_m angle between B field and rotation axis
    # t [s], omega [1/s]
    
#    xNew = zero(x)
#    for i in 1:length(x[:,1])
#        xNew[i, :] .= rotate_y(θR) * x[i, :]
#    end
    x0_pl = nothing
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
        x0_pl = [r θ ϕ]
    else
        x0_pl = x
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
    
    g_tt, g_rr, g_thth, g_pp = g_schwartz(x0_pl, Mass_NS)
    B_mu = funcB(x0_pl, t, θm, ω, B0, rNS; Mass_NS=Mass_NS, sphericalX=true, flat=flat, return_comp=-1, include_metric=true)
    Bz = B_mu[:, 1] .* sqrt.(g_rr) .* cos.(θ) .- B_mu[:, 2] .* sqrt.(g_thth) .* sin.(θ)
    
    
    nelec = ((-2.0 .* ω .* Bz[1]) ./ sqrt.(4 .* π ./ 137) .* 1.95e-2 .* hbar) ; # eV^3
    if rho_twist
        x0_C = nothing
        if !sphericalX
            x0_C = x
        else
            x0_C = [r .* sin.(θ) .* cos(ϕ) r .* sin.(θ) .* sin(ϕ) r .* cos.(θ)]
        end
        
        if time_deriv
            function wrapperB(x; indx=1)
                return funcB(x[:, 1:3], x[:, 4], θm, ω, B0, rNS; Mass_NS=0.0, sphericalX=false, include_metric=false, return_comp=indx, sphere_out=false)[1]
            end
            xIn = [r .* sin.(θ) .* cos(ϕ) r .* sin.(θ) .* sin(ϕ) r .* cos.(θ) t.value]
        
            HBX = hessian(x -> wrapperB(x, indx=1), xIn)
            gradB_V = grad(funcB(seed(x0_C), t.value, θm, ω, B0, rNS; Mass_NS=0.0, sphericalX=false, include_metric=false, return_comp=1, sphere_out=false))
            
            dB_xx = Dual(gradB_V[1], HBX[end, 1])
            dB_xy = Dual(gradB_V[2], HBX[end, 2])
            dB_xz = Dual(gradB_V[3], HBX[end, 3])
                
           
        else
            gradB = grad(funcB(seed(x0_C), t, θm, ω, B0, rNS; Mass_NS=0.0, sphericalX=false, include_metric=false, return_comp=1, sphere_out=false))
            
            if isnan(gradB[1])
                # avoiding Nan arising from jump in phi coordinate at [0.0 0.0]
                x0_C .+= [1e-5 1e-5 1e-5]
                gradB = grad(funcB(seed(x0_C), t, θm, ω, B0, rNS; Mass_NS=0.0, sphericalX=false, include_metric=false, return_comp=1, sphere_out=false))
            end
            dB_xx = gradB[1]
            dB_xy = gradB[2]
            dB_xz = gradB[3]
        end
        

        
        if time_deriv
            HBX = hessian(x -> wrapperB(x, indx=2), xIn)
            gradB_V = grad(funcB(seed(x0_C), t.value, θm, ω, B0, rNS; Mass_NS=0.0, sphericalX=false, include_metric=false, return_comp=2, sphere_out=false))
            
            dB_yx = Dual(gradB_V[1], HBX[end, 1])
            dB_yy = Dual(gradB_V[2], HBX[end, 2])
            dB_yz = Dual(gradB_V[3], HBX[end, 3])
                
        else
            gradB = grad(funcB(seed(x0_C), t, θm, ω, B0, rNS; Mass_NS=0.0, sphericalX=false, include_metric=false, return_comp=2, sphere_out=false))
            dB_yx = gradB[1]
            dB_yy = gradB[2]
            dB_yz = gradB[3]
        end
        
        
        if time_deriv
            HBX = hessian(x -> wrapperB(x, indx=3), xIn)
            gradB_V = grad(funcB(seed(x0_C), t.value, θm, ω, B0, rNS; Mass_NS=0.0, sphericalX=false, include_metric=false, return_comp=3, sphere_out=false))
           
            dB_zx = Dual(gradB_V[1], HBX[end, 1])
            dB_zy = Dual(gradB_V[2], HBX[end, 2])
            dB_zz = Dual(gradB_V[3], HBX[end, 3])
                
        else
            gradB = grad(funcB(seed(x0_C), t, θm, ω, B0, rNS; Mass_NS=0.0, sphericalX=false, include_metric=false, return_comp=3, sphere_out=false))
            dB_zx = gradB[1]
            dB_zy = gradB[2]
            dB_zz = gradB[3]
        end
        
        
        mag_curlB = sqrt.((dB_zy .- dB_yz).^2 .+ (dB_xz .- dB_zx).^2 .+ (dB_yx .- dB_xy).^2)
        nelec += mag_curlB ./ sqrt.(4 .* π ./ 137) .*  (c_km .* hbar)
    end
    ωp = sqrt.(4 .* π .* abs.(nelec) ./ 137 ./ 5.0e5);
    
    if null_fill > 0.0
          
        nelecP = ((-2.0 .* ω .* B0 * cos.(θm)) ./ sqrt.(4 .* π ./ 137) .* 1.95e-2 .* hbar) ; # eV^3
        ωpPole = sqrt.(4 .* π .* abs.(nelecP) ./ 137 ./ 5.0e5);
        rmax = rNS .* (ωpPole ./ Mass_a).^(2.0 ./ 3.0)
        
        n1 = ((3.0 .+ 0.5 * (tanh.(B_DQ .- 2.0) .+ 1.0)) ./ 2.0)
        # n2 = null_fill
        # delX = 0.01 .* n1 ./ n2 .* (ωpPole ./ Mass_a).^(n2 ./ n1 )
        # ωp *= (1.0 .+ delX .* (rNS ./ r).^null_fill);
        
        ωp += ωpPole .* (rNS ./ r).^n1 .* exp.(- (r .- rmax .* null_fill) ./ (0.1 .* rmax))
        # print("here \t", r, "\t", Mass_a, "\t", rNS, "\t", rmax, "\t", ωpPole, "\n")
    end
    
    if (reflect_LFL)&&(θm==0.0)
        Lline = r ./ sin.(θ).^2
        L_LC = c_km ./ ω
        if isinf.(Lline)
            Lline = L_LC .* 1e3
        end
        # print(ωp, "\t", Lline, "\t", L_LC, "\t", reflect_norm .* (rNS ./ r).^(3.0 ./ 2.0), "\n")
        ωp += reflect_norm .* (rNS ./ r).^(3.0 ./ 2.0) .* exp.(- abs.(Lline .- L_LC) ./ (0.1 .* Lline))
        
    end
    
    if dead
        ωp *= exp.(- r ./ dead_rmax)
    end
    
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


function ellipse_plasma(x, t, θm, ω, B0, rNS; zeroIn=true, sphericalX=true)
    
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
    omegaP_pole = sqrt.(4.0 .* pi .* a_fs .* (ω * B0 * cos.(θm) / e_charge * G_to_eV2 * hbar) / m_elec)
    r_hold = rNS * sqrt.(2.0) ./ sqrt.(1.0 .+ 2 .^(2.0/3.0) .- (-1.0 .+ 2.0 .^(2.0/3.0)) .* cos.(2.0 .* θ))
    ωp = omegaP_pole .* (r_hold / r).^(3.0/2.0)
    
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

function Spherical_Plasma_Test(x, t, θm, ω, B0, rNS; zeroIn=true, sphericalX=true)
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
    ωp = 1e-6 .* (r ./ (5.0 .* rNS)).^(-3.0 ./ 2.0)
    return ωp
end

# dipolar magnetic field
function B_dipolar(x, t, θm, ω, B0, rNS; Mass_NS=1.0, flat=false, sphericalX=false, return_comp=-1, include_metric=true, sphere_out=true)
    if flat
        Mass_NS = 0.0
    end
    if !sphericalX
        
        r = sqrt.(sum(x .^2 , dims=2))
        ϕ = atan.(view(x, :, 2), view(x, :, 1))
        θ = acos.(view(x, :, 3)./ r)
    else
        if length(size(x)) > 1
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
    
    x0_pl = [r θ ϕ]
    g_tt, g_rr, g_thth, g_pp = g_schwartz(x0_pl, Mass_NS)
    if return_comp == -1
        return [Br ./ sqrt.(g_rr) Btheta ./ sqrt.(g_thth)  Bphi ./ sqrt.(g_pp)] # lower
    elseif return_comp == 0
        return sqrt.(Br.^2 .+ Btheta.^2 .+ Bphi.^2) * 1.95e-2
    elseif return_comp == 1
        return Br ./ sqrt.(g_rr) .* g_rr * 1.95e-2 # this is for d_mu B^i
    elseif return_comp == 2
        return Btheta ./ sqrt.(g_thth) .* g_thth * 1.95e-2
    elseif return_comp == 3
        return Bphi ./ sqrt.(g_pp) .* g_pp * 1.95e-2
    end
end

function B_Di_Quad(x, t, θmD, θmQ, phiQ, ω, B0, B_DQ, rNS; Mass_NS=1.0, flat=false, sphericalX=false, return_comp=-1, include_metric=true, sphere_out=true)
    if flat
        Mass_NS = 0.0
    end
    if !sphericalX
        r = sqrt.(sum(x.^2))
        ϕ = atan.(x[2], x[1])
        θ = acos.(x[3] ./ r)
#        r = sqrt.(sum(xNew .^2 , dims=2))
#        ϕ = atan.(view(xNew, :, 2), view(xNew, :, 1))
#        θ = acos.(view(xNew, :, 3)./ r)
    else
        
#        if length(size(x)) > 2
#
#            r = view(x, :, 1)
#            θ = view(x, :, 2)
#            ϕ = view(x, :, 3)
#        else
            
        r = x[1]
        θ = x[2]
        ϕ = x[3]
#        end
    end
    
    # dipolar
    ψ = ϕ .- ω .* t

   
    Bnorm = B0 .* (rNS ./ r).^3 ./ 2
    Br = 2 .* Bnorm .* (cos.(θmD) .* cos.(θ) .+ sin.(θmD) .* sin.(θ) .* cos.(ψ))
    Btheta = Bnorm .* (cos.(θmD) .* sin.(θ) .- sin.(θmD) .* cos.(θ) .* cos.(ψ))
    Bphi = Bnorm .* sin.(θmD) .* sin.(ψ)
    
    
    # quadrupolar
    B_Q = B0 .* B_DQ .* (rNS ./ r).^4 ./ 2
    ψQ = ϕ .- ω .* t .+ phiQ
    Br += B_Q .* ( (3 .* cos.(θ).^2 .- 1.0) .* cos.(θmQ) .+ 3.0 .* sin.(θmQ) .* sin.(θ) .* cos.(θ) .* cos.(ψQ))
    Btheta += B_Q .* (cos.(θmQ) .* sin.(2 .* θ) .- sin.(θmQ) .* cos.(2 .* θ) .* cos.(ψQ) )
    Bphi += B_Q .* sin.(θmQ) .* sin.(ψQ) .* cos(θ)

    
    x0_pl = [r θ ϕ]
    g_tt, g_rr, g_thth, g_pp = g_schwartz(x0_pl, Mass_NS)
    if return_comp == -1
        if include_metric
            # print("TEST ! \t ", g_rr, "\n") BROKEN HERE
            return [Br ./ sqrt.(g_rr) Btheta ./ sqrt.(g_thth)  Bphi ./ sqrt.(g_pp)] # lower
        else
            return [Br Btheta Bphi] # lower
        end
    elseif return_comp == 0
        return sqrt.(Br.^2 .+ Btheta.^2 .+ Bphi.^2) * 1.95e-2
    elseif return_comp == 1
        
        if sphere_out
            if include_metric
                return Br ./ sqrt.(g_rr) .* g_rr * 1.95e-2 # this is for d_mu B^i
            else
                return Br .* 1.95e-2 # this is for d_mu B^i
            end
        else
            return (Br .* sin.(θ) .* cos.(ϕ) .+ Btheta .* cos.(θ) .* cos.(ϕ) .- Bphi .* sin.(ϕ)) .* 1.95e-2
        end
    elseif return_comp == 2
        if sphere_out
            if include_metric
                return Btheta ./ sqrt.(g_thth) .* g_thth * 1.95e-2
            else
                return Btheta .* 1.95e-2
            end
        else
            return (Br .* sin.(θ) .* sin.(ϕ) .+ Btheta .* cos.(θ) .* sin.(ϕ) .+ Bphi .* cos.(ϕ)) .* 1.95e-2
        end
    elseif return_comp == 3
        if sphere_out
            if include_metric
                return Bphi ./ sqrt.(g_pp) .* g_pp * 1.95e-2
            else
                return Bphi .* 1.95e-2
            end
        else
            return (Br .* cos.(θ) .- Btheta .* sin.(θ)) .* 1.95e-2
        end
    end
end


function Magnetar_plasma(x, t, θm, ω, B0, rNS; lambdaM=1.0, psi=0.2, zeroIn=true, sphericalX=true)
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
    
    nelec = 7.0e15 .* (c_cm .* hbar).^3 .* sin.(θ).^2 .* lambdaM .* (psi ./ 0.2) .* (B0 ./ 2e14) .* (rNS ./ r).^4
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

function rotate_y(θ)
    Ry = [cos.(θ) 0.0 sin.(θ); 0.0 1.0 0.0; -sin.(θ) 0.0 cos.(θ)]
    return Ry
end

function g_schwartz(x0, Mass_NS; rNS=10.0, flat=false)
    # (1 - r_s / r)
    # notation (-,+,+,+), defined upper index g^mu^nu
    
    if size(x0)[1] > 1
        r = x0[:,1]
        rs = ones(eltype(r), size(r)) .* 2 * GNew .* Mass_NS ./ c_km.^2
        rs[r .<= rNS] .*= (r[r .<= rNS] ./ rNS).^3
        sin_theta = sin.(x0[:,2])
        
    else
        
        rs = 2 * GNew .* Mass_NS ./ c_km.^2
        r = x0[1]
        if r <= rNS
            
            rs = 2 * GNew .* Mass_NS ./ c_km.^2 .* (r ./ rNS).^3
        end
        sin_theta = sin.(x0[2])
    end
    
    g_tt = -1.0 ./ (1.0 .- rs ./ r);
    g_rr = (1.0 .- rs ./ r);
    if flat
        g_rr ./= g_rr
        g_tt .*= g_rr
    end
    g_thth = 1.0 ./ r.^2; # 1/km^2
    g_pp = 1.0 ./ (r.^2 .* sin_theta.^2); # 1/km^2
    
    
    return g_tt, abs.(g_rr), g_thth, g_pp
    
end

function spatial_dot(vec1, vec2, x0_pl, Mass_NS)
    # assumes both vectors v_mu
    g_tt, g_rr, g_thth, g_pp = g_schwartz(x0_pl, Mass_NS);
    ntrajs = length(g_tt)
    out_v = zeros(ntrajs)
    for i in 1:ntrajs
        out_v[i] = (g_rr[i] .* vec1[i, 1] .* vec2[i, 1] .+ g_thth[i] .* vec1[i, 2] .* vec2[i, 2] .+ g_pp[i] .* vec1[i, 3] .* vec2[i, 3])
    end
    return out_v
end
