using ArgParse
using SpecialFunctions
using LinearAlgebra
using NPZ
using Dates
using Statistics
using Base
using HDF5
include("Population_Fitter.jl")


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
    
        "--Nsamples"
            arg_type = Int
            default = 10000000

        "--Nruns"
            arg_type = Int
            default = 5
            
        "--numwalkers"
            arg_type = Int
            default = 10
            
        "--minimizeIt"
            arg_type = Bool
            default = false
            
        "--run_analysis"
            arg_type = Bool
            default = true
            
        "--run_plot_data"
            arg_type = Bool
            default = false
            
        "--run_Combine"
            arg_type = Bool
            default = false

        "--tau_ohmic"
            arg_type = Float64
            default = 10.0e6

        "--max_T_f"
            arg_type = Float64
            default = 5.0

        "--fileName"
            arg_type = String
            default = "Test_Run"

        "--run_magnetars"
            arg_type = Bool
            default = false
            
        "--kill_dead"
            arg_type = Bool
            default = false
        
        "--Pmin"
            arg_type = Float64
            default = 0.02
            
        "--Pmax"
            arg_type = Float64
            default = 0.75
            
        "--Bmin"
            arg_type = Float64
            default = 3e12
            
        "--Bmax"
            arg_type = Float64
            default = 4e13
            
        "--sigP_min"
            arg_type = Float64
            default = 0.05
            
        "--sigP_max"
            arg_type = Float64
            default = 0.5
            
        "--sigB_min"
            arg_type = Float64
            default = 0.1
            
        "--sigB_max"
            arg_type = Float64
            default = 1.0
            
        "--Npts_P"
            arg_type = Int
            default = 5
            
        "--Npts_B"
            arg_type = Int
            default = 5

        "--NPts_Psig"
            arg_type = Int
            default = 5
            
        "--NPts_Bsig"
            arg_type = Int
            default = 5

    end

    return parse_args(s)
end

parsed_args = parse_commandline()

minimizeIt = parsed_args["minimizeIt"];
run_analysis = parsed_args["run_analysis"];
run_plot_data = parsed_args["run_plot_data"];
tau_ohmic = parsed_args["tau_ohmic"];
max_T_f = parsed_args["max_T_f"];
fileName = parsed_args["fileName"];
run_magnetars = parsed_args["run_magnetars"];
kill_dead = parsed_args["kill_dead"];
Pmin = parsed_args["Pmin"];
Pmax = parsed_args["Pmax"];
Bmin = parsed_args["Bmin"];
Bmax = parsed_args["Bmax"];
sigP_min = parsed_args["sigP_min"];
sigP_max = parsed_args["sigP_max"];
sigB_min = parsed_args["sigB_min"];
sigB_max = parsed_args["sigB_max"];
Npts_P = parsed_args["Npts_P"];
Npts_B = parsed_args["Npts_B"];
NPts_Psig = parsed_args["NPts_Psig"];
NPts_Bsig = parsed_args["NPts_Bsig"];
Nruns = parsed_args["Nruns"];
numwalkers = parsed_args["numwalkers"]
Nsamples = parsed_args["Nsamples"];


if minimizeIt
    temp=false
else
    temp=true
end


time0=Dates.now()

if run_analysis == true
    @inbounds @fastmath main(run_analysis, run_plot_data, tau_ohmic; Nsamples=Nsamples, max_T_f=max_T_f, fileName=fileName, xIn=[0.05, log10.(1.4e13), 0.05, 0.65, 0.0], run_magnetars=run_magnetars, kill_dead=kill_dead,  Pmin=Pmin, Pmax=Pmax, Bmin=Bmin, Bmax=Bmax, sigP_min=sigP_min, sigP_max=sigP_max, sigB_min=sigB_min, sigB_max=sigB_max, Npts_P=Npts_P, Npts_B=Npts_B, NPts_Psig=NPts_Psig, NPts_Bsig=NPts_Bsig, temp=temp, minimizeIt=minimizeIt, numwalkers=numwalkers, Nruns=Nruns);
end


function combine_files(run_analysis, run_plot_data, tau_ohmic; max_T_f=5.0, fileName="Test_Run", xIn=[0.05, log10.(1.4e13), 0.05, 0.65, 0.0], run_magnetars=false, kill_dead=false,  Pmin=0.05, Pmax=0.75, Bmin=1e12, Bmax=5e13, sigP_min=0.05, sigP_max=0.4, sigB_min=0.1, sigB_max=1.2, Npts_P=5, Npts_B=5, NPts_Psig=5, NPts_Bsig=5, temp=false, Nruns=2)
   
    fileL = [];
 
    for i = 0:(Nruns-1)
        fileN = "temp/"*fileName*string(i)*".dat"
        push!(fileL, fileN);
    end
    
    fileFull = File_Name_Out(run_analysis, run_plot_data, tau_ohmic; max_T_f=max_T_f, fileName=fileName, run_magnetars=run_magnetars, kill_dead=kill_dead,  Pmin=Pmin, Pmax=Pmax, Bmin=Bmin, Bmax=Bmax, sigP_min=sigP_min, sigP_max=sigP_max, sigB_min=sigB_min, sigB_max=sigB_max, Npts_P=Npts_P, Npts_B=Npts_B, NPts_Psig=NPts_Psig, NPts_Bsig=NPts_Bsig)
   
   outputTable = nothing
    for i in 1:length(fileL)
        tempH = open(readdlm, fileL[i])
        if i == 1
            outputTable = tempH
        else
            outputTable = vcat(outputTable, tempH)
        end
    end

    writedlm("output_fits/"*fileFull, outputTable)

    for i = 1:Nruns
        Base.Filesystem.rm(fileL[i])
    end
    
end

if parsed_args["run_Combine"] == true
    combine_files(run_analysis, run_plot_data, tau_ohmic; max_T_f=max_T_f, fileName=fileName, xIn=[0.05, log10.(1.4e13), 0.05, 0.65, 0.0], run_magnetars=run_magnetars, kill_dead=kill_dead,  Pmin=Pmin, Pmax=Pmax, Bmin=Bmin, Bmax=Bmax, sigP_min=sigP_min, sigP_max=sigP_max, sigB_min=sigB_min, sigB_max=sigB_max, Npts_P=Npts_P, Npts_B=Npts_B, NPts_Psig=NPts_Psig, NPts_Bsig=NPts_Bsig, temp=false, Nruns=Nruns);
end


time1=Dates.now()
print("\n\n Run time: ", time1-time0, "\n")

