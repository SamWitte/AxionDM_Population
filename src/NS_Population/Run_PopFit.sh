
Fname="Pop_Tau10y_DeadK_"

Npts_P=5
Npts_B=5
Pmin=0.02
Pmax=0.7
Bmin=5e12
Bmax=3e13
Nsamps=1000000

sigP_min=0.1
dsP=0.1
sigB_min=0.2
dsB=0.2
NPts_Psig=5
NPts_Bsig=5

tau_ohmic=10.0e6
max_T_f=10.0
run_magnetars=false
kill_dead=true

julia Gen_pop_fit.jl --Bmax $Bmax  --Bmin $Bmin --Pmax $Pmax --Pmin $Pmin --sigP_min $tempSp --sigP_max $tempSp --sigB_min $tempSb --sigB_max $tempSb --fileName $Fname --run_analysis false --run_Combine false --run_plot_data true --tau_ohmic $tau_ohmic --max_T_f $max_T_f --run_magnetars $run_magnetars --kill_dead $kill_dead






