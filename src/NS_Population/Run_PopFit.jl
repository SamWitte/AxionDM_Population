include("Population_Fitter.jl")

run_analysis = true
run_plot_data = false

# tau_ohmic = 10.0e6
# max_T_f = 10.0
tau_ohmic = 1.0e7
max_T_f = 10.0
fileName = "TestSamples_secondB_"
# xIN = [0.28, 13.44, 0.11, 0.86, 0.0]
xIN = [-0.2, 12.70, 0.4, 0.5]

run_magnetars=false
kill_dead=true
minimizeIt=false
Nsamples=100000
gauss_approx=true


main(run_analysis, run_plot_data, tau_ohmic; Nsamples=Nsamples, max_T_f=max_T_f, fileName=fileName, xIn=xIN, run_magnetars=run_magnetars, kill_dead=kill_dead, minimizeIt=false, gauss_approx=gauss_approx);
