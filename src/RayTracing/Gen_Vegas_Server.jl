using ArgParse
using NPZ
using Dates
using Statistics
using Random
using Glob
using DelimitedFiles
include("Constants.jl")
include("Vegas_Sampling.jl")
include("environment.jl")


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--thetaN"
            arg_type = Int
            default = 5
            
        "--theta_target"
            arg_type = Float64
            default = 1.0
            
        "--theta_err"
            arg_type = Float64
            default = 0.05
            
        "--c_phi"
            arg_type = Bool
            default = false
            
        "--phi_target"
            arg_type = Float64
            default = 1.0
            
        "--phi_err"
            arg_type = Float64
            default = 0.05
            
        "--ThetaM"
            arg_type = Float64
            default = 0.0
            
        "--MassA"
            arg_type = Float64
            default = 1e-5
            
        "--AxG"
            arg_type = Float64
            default = 1e-12
        
        "--B0"
            arg_type = Float64
            default = 1e14

        "--rotW"
            arg_type = Float64
            default = 1.0
            
        "--rhoDM"
            arg_type = Float64
            default = 1.0
            
        "--vmean_ax"
            arg_type = Float64
            default = 220.0

        "--rNS"
            arg_type = Float64
            default = 10.0
            
        "--Mass_NS"
            arg_type = Float64
            default = 1.0
        
            
        "--Flat"
            arg_type = Bool
            default = false

        "--Iso"
            arg_type = Bool
            default = false
            
        "--CLen_Scale"
            arg_type = Bool
            default = false
            
        "--add_tau"
            arg_type = Bool
            default = false
        
        "--Thick"
            arg_type = Bool
            default = true
        
        "--debug"
            arg_type = Bool
            default = true
            
        "--maxitrs"
            arg_type = Int
            default = 10
            
        "--nbins"
            arg_type = Int
            default = 2
            
        "--ncalls"
            arg_type = Int
            default = 1000

        "--ftag"
            arg_type = String
            default = ""
            
        "--output_dir"
            arg_type = String
            default = ""
            
        "--delta_on_v"
            arg_type = Bool
            default = false
            
        "--compute_point"
            arg_type = Bool
            default = true
            
        "--return_width"
            arg_type = Bool
            default = true
            
        ########### Magneto Model ###########
        "--plasma_mod"
            arg_type = String
            default = "GJ"
        "--mag_mod"
            arg_type = String
            default = "Dipole"
            
        "--B_DQ"
            arg_type = Float64
            default = 0.0
        "--ThetaQ"
            arg_type = Float64
            default = 0.0
        "--PhiQ"
            arg_type = Float64
            default = 0.0
            
        "--null_fill"
            arg_type = Float64
            default = 0.0
        "--reflect_LFL"
            arg_type = Bool
            default = false
        
        "--dead"
            arg_type = Bool
            default = false
        "--dead_rmax"
            arg_type = Float64
            default = 30.0
            
        "--lambdaM"
            arg_type = Float64
            default = 1.0
        "--psi"
            arg_type = Float64
            default = 0.0

    
        
    end
    return parse_args(s)
end

prsd_args = parse_commandline()


## Target params
# theta_target = LinRange(0.1, pi/2.0, prsd_args["thetaN"])
theta_target = [prsd_args["theta_target"]]
theta_err = prsd_args["theta_err"]
constrain_phi = prsd_args["c_phi"]
phi_target = prsd_args["phi_target"]
phi_err = prsd_args["phi_err"]

## Standard params
Mass_a = prsd_args["MassA"]; # eV
Ax_g = prsd_args["AxG"]; # 1/GeV
θm = prsd_args["ThetaM"]; # rad
ωPul = prsd_args["rotW"]; # 1/s
B0 = prsd_args["B0"]; # G
rNS = prsd_args["rNS"]; # km
Mass_NS = prsd_args["Mass_NS"]; # solar mass
rho_DM=prsd_args["rhoDM"] # GeV/cm^3
vmean_ax=prsd_args["vmean_ax"]
CLen_Scale = prsd_args["CLen_Scale"] # if true, perform cut due to de-phasing
fix_time = 0.0; # eval at fixed time = 0?
file_tag = "_";  #
ode_err = 1e-6;
vNS = [0 0 0]; # relative neutron star velocity
flat = prsd_args["Flat"]; # flat space or schwartzchild
isotropic = prsd_args["Iso"]; # default is anisotropic
melrose = true;
thick_surface = prsd_args["Thick"]
delta_on_v = prsd_args["delta_on_v"]
CLen_Scale = prsd_args["CLen_Scale"] # if true, perform cut due to de-phasing
add_tau = prsd_args["add_tau"] # dont use right now, under construction

# iseed = 1000

# plasma_mod = "GJ"
# mag_mod = "Dipole"
plasma_mod = prsd_args["plasma_mod"]
mag_mod = prsd_args["mag_mod"]
B_DQ = prsd_args["B_DQ"]
θmQ = prsd_args["ThetaQ"]
phiQ = prsd_args["PhiQ"]
lambdaM = prsd_args["lambdaM"]
psi = prsd_args["psi"]
null_fill = prsd_args["null_fill"]
reflect_LFL = prsd_args["reflect_LFL"]
dead = prsd_args["dead"]
dead_rmax = prsd_args["dead_rmax"]

print("Using Model: \n")
print("Plasma: ", plasma_mod, "\n")
print("Magnetic Field: ", mag_mod, "\n\n")

## Vegas params
maxiter=prsd_args["maxitrs"]
nbins=prsd_args["nbins"]
ncalls=prsd_args["ncalls"]
debug=prsd_args["debug"]
return_width = prsd_args["return_width"]

### sve
sve_output=true
output_arr = zeros(length(theta_target), 4)
compute_point = prsd_args["compute_point"]

time0=Dates.now()
# @inbounds @fastmath
print("number threads: ", Threads.nthreads(), "\n")


Threads.@threads for i = 1:length(theta_target)
    finalV, std_v, chi2_v, meanE, stdE = Vegas_sampler(theta_target[i], theta_err, Mass_a, θm, ωPul, B0, rNS; return_width=return_width, maxiter=maxiter, debug=debug, nbins=nbins, ncalls=ncalls, constrain_phi=constrain_phi, phi_target=phi_target, phi_err=phi_err, fix_time=fix_time, rho_DM=rho_DM, vmean_ax=vmean_ax, Ax_g=Ax_g, Mass_NS=Mass_NS, thick_surface=thick_surface, flat=flat, isotropic=isotropic, melrose=melrose, ode_err=ode_err, add_tau=add_tau, CLen_Scale=CLen_Scale, plasma_mod=plasma_mod, mag_mod=mag_mod, lambdaM=lambdaM, psi=psi, delta_on_v=delta_on_v, θmQ=θmQ, phiQ=phiQ, B_DQ=B_DQ, null_fill=null_fill, reflect_LFL=reflect_LFL, dead=dead, dead_rmax=dead_rmax)
    output_arr[i, :] = [theta_target[i] finalV meanE stdE]
end


open(prsd_args["output_dir"]*"/"*prsd_args["ftag"]*"_Theta_"*string(prsd_args["theta_target"])*".txt", "w") do io
    writedlm(io, output_arr)

            
time1=Dates.now()
print("\n")
print("time diff: ", time1-time0)
print("\n")

