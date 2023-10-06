using ArgParse
using SpecialFunctions
using LinearAlgebra
using NPZ
using Dates
using Statistics
using Base
using HDF5
include("Constants.jl")
include("environment.jl")
include("RayTracer_Fast_GR_2.jl")

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--ThetaM"
            arg_type = Float64
            default = 0.2

        "--Nts"
            arg_type = Int
            default = 10000

        "--ftag"
            arg_type = String
            default = ""

        "--rotW"
            arg_type = Float64
            default = 1.0

        "--MassA"
            arg_type = Float64
            default = 1e-6
            
        "--AxG"
            arg_type = Float64
            default = 1e-12
        
        "--B0"
            arg_type = Float64
            default = 1e14
            
        "--run_RT"
            arg_type = Int
            default = 1
            
        "--run_Combine"
            arg_type = Int
            default = 0
            
        "--side_runs"
            arg_type = Int
            default = 0
            
        "--rNS"
            arg_type = Float64
            default = 10.0
            
        "--Mass_NS"
            arg_type = Float64
            default = 1.0
            
        "--vNS_x"
            arg_type = Float64
            default = 0.0
            
        "--vNS_y"
            arg_type = Float64
            default = 0.0
            
        "--vNS_z"
            arg_type = Float64
            default = 0.0
            
        "--Flat"
            arg_type = Bool
            default = false

        "--Iso"
            arg_type = Bool
            default = false
            
        "--CP_one_D"
            arg_type = Bool
            default = false

        "--add_tau"
            arg_type = Bool
            default = false
        
        "--Thick"
            arg_type = Bool
            default = true
            
        "--delta_on_v"
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

########### Saving ###########
        "--sve_mode"
            arg_type = Int
            default = 1

    end

    return parse_args(s)
end

parsed_args = parse_commandline()

Mass_a = parsed_args["MassA"]; # eV
Ax_g = parsed_args["AxG"]; # 1/GeV

θm = parsed_args["ThetaM"]; # rad
ωPul = parsed_args["rotW"]; # 1/s
B0 = parsed_args["B0"]; # G
rNS = parsed_args["rNS"]; # km
Mass_NS = parsed_args["Mass_NS"]; # solar mass
gammaF = [1.0, 1.0]

Ntajs = parsed_args["Nts"];
batchSize = 1;
add_tau = parsed_args["add_tau"] # dont use right now, under construction
fix_time = 0.0; # eval at fixed time = 0?
file_tag = parsed_args["ftag"];  # if you dont want to cut on Lc "_NoCutLc_";
ode_err = 1e-6;
ntimes = 4
vNS = [parsed_args["vNS_x"] parsed_args["vNS_y"] parsed_args["vNS_z"]]; # relative neutron star velocity
flat = parsed_args["Flat"]; # flat space or schwartzchild
isotropic = parsed_args["Iso"]; # default is anisotropic
melrose = true;
CP_one_D = parsed_args["CP_one_D"]; # use 1D conversion?
thick_surface = parsed_args["Thick"]
delta_on_v = parsed_args["delta_on_v"]
sve_mode = parsed_args["sve_mode"]

plasma_mod = parsed_args["plasma_mod"]
mag_mod = parsed_args["mag_mod"]
B_DQ = parsed_args["B_DQ"]
θmQ = parsed_args["ThetaQ"]
phiQ = parsed_args["PhiQ"]
lambdaM = parsed_args["lambdaM"]
psi = parsed_args["psi"]
null_fill = parsed_args["null_fill"]
reflect_LFL = parsed_args["reflect_LFL"]
dead = parsed_args["dead"]
dead_rmax = parsed_args["dead_rmax"]


print("Using Model: \n")
print("Plasma: ", plasma_mod, "\n")
print("Magnetic Field: ", mag_mod, "\n")
print("Charge Mult: ", lambdaM, "\n")
print("Twist Angle: ", psi, "\n\n")

print("Parameters: \n")
print("Axion Mass: ", Mass_a, "\n")
print("Axion coupling: ", Ax_g, "\n\n")
print("Misalignment angle: ", θm, "\n")
print("Pular angular velocity: ", ωPul, "\n")
print("Pulsar B-field: ", B0, "\n")
print("Radius NS: ", rNS, "\n")
print("Mass NS: ", Mass_NS, "\n\n")
print("Num. trajs. to run: ", Ntajs, "\n")
print("Batchsize: ", batchSize, "\n")
print("Include optical depth: ", add_tau, "\n")
print("Global time at source: ", fix_time, "\n")
print("Times saveat: ", ntimes, "\n")
print("Flat: ", flat, "\n")
print("Isotropic: ", isotropic, "\n")
print("Thick surface: ", thick_surface, "\n")

if parsed_args["run_Combine"] == 0
    fileT_check=file_tag[1:end-1]
else
    fileT_check=file_tag
end
fileN_TEST = File_Name_Out(Mass_a, Ax_g, Mass_NS, rNS, vNS, B0, 2 * pi / ωPul, θm, null_fill, Int(Ntajs*parsed_args["side_runs"]); file_tag=fileT_check, mag_mod=mag_mod, B_DQ=B_DQ, θQ=θmQ, reflect=reflect_LFL, dead=dead, dead_rmax=dead_rmax)

if isfile(fileN_TEST)
    print("File already exists. \n")
    parsed_args["run_RT"] = 0
    parsed_args["run_Combine"] = 0
end

time0=Dates.now()

if parsed_args["run_RT"] == 1
    @inbounds @fastmath main_runner(Mass_a, Ax_g, θm, ωPul, B0, rNS, Mass_NS, Ntajs; flat=flat, isotropic=isotropic, melrose=melrose, ode_err=ode_err, fix_time=fix_time, add_tau=add_tau, file_tag=file_tag, ntimes=ntimes, v_NS=vNS, thick_surface=thick_surface, sve_mode=sve_mode, plasma_mod=plasma_mod, mag_mod=mag_mod, lambdaM=lambdaM, psi=psi, delta_on_v=delta_on_v, θmQ=θmQ, phiQ=phiQ, B_DQ=B_DQ, null_fill=null_fill, reflect_LFL=reflect_LFL, dead=dead, dead_rmax=dead_rmax);
end


function combine_files(Mass_a, Ax_g, θm, ωPul, B0, Ntajs, Nruns, ode_err, fix_time, file_tag, ntimes, v_NS, Mass_NS, rNS; null_fill=0.0, mag_mod="Dipole", B_DQ=0.0, θQ=0.0, reflect_LFL=false, dead=false, dead_rmax=30.0)
   
    fileL = String[];
    
    P = 2 .* pi ./ ωPul
    
    
    for i = 0:(Nruns-1)
        fileN = File_Name_Out(Mass_a, Ax_g, Mass_NS, rNS, v_NS, B0, P, θm, null_fill, ntimes; file_tag=file_tag, indx=i, mag_mod=mag_mod, B_DQ=B_DQ, θQ=θmQ, reflect=reflect_LFL, dead=dead, dead_rmax=dead_rmax)
    
        push!(fileL, fileN);
    end
    
    fileN = File_Name_Out(Mass_a, Ax_g, Mass_NS, rNS, v_NS, B0, P, θm, null_fill, Int(Ntajs*Nruns); file_tag=file_tag, mag_mod=mag_mod, B_DQ=B_DQ, θQ=θmQ, reflect=reflect_LFL, dead=dead, dead_rmax=dead_rmax)
   
    fid = h5open(fileN, "w")
    
    
    file_first = h5open(fileL[1], "r")
    keyNames = nothing;
    try
        keyNames = names(file_first)
    catch
        keyNames = keys(file_first)
    end
    for i in 1:length(keyNames)
        hold_vals = read(file_first[keyNames[i]]);
        for j in 2:length(fileL)
            file_temp = h5open(fileL[j], "r")
            hold_vals = [hold_vals; read(file_temp[keyNames[i]])];
            close(file_temp)
        end
        if keyNames[i] == "weights"
            hold_vals ./= Nruns
        end
        
        fid[keyNames[i]] = hold_vals
    end
    close(fid)
    close(file_first)


    for i = 1:Nruns
        Base.Filesystem.rm(fileL[i])
    end
    
end

if parsed_args["run_Combine"] == 1
    combine_files(Mass_a, Ax_g, θm, ωPul, B0, Ntajs, parsed_args["side_runs"], ode_err, fix_time, file_tag, ntimes, vNS, Mass_NS, rNS, null_fill=null_fill, mag_mod=mag_mod, B_DQ=B_DQ, θQ=θmQ, reflect_LFL=reflect_LFL, dead=dead, dead_rmax=dead_rmax);
end


time1=Dates.now()
print("\n\n Run time: ", time1-time0, "\n")

