#!/bin/bash
source /mnt/zfsusers/switte/.bashrc

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

tau_ohmic=10.0
max_T_f=5.0
run_magnetars=false
kill_dead=true

declare -i memPerjob
memPerjob=$((SLURM_MEM_PER_CPU))

echo "Testing ...." $SLURM_NTASKS
cntTot=0
for ((i = 0; i < $NPts_Psig ; i++)); do
    tempSp=$( echo "$sigP_min + $i*$dsP" | bc )
    for ((j = 0; j < $NPts_Bsig ; j++)); do
        tempSb=$( echo "$sigB_min + $j*$dsB" | bc )
        srun --ntasks=1 --exclusive --cpus-per-task=1 --mem=$memPerjob julia --threads 1 Gen_pop_fit.jl --NPts_Psig 1 --NPts_Bsig 1 --Npts_B $Npts_B --Npts_P $Npts_P --Bmax $Bmax  --Bmin $Bmin --Pmax $Pmax --Pmin $Pmin --sigP_min $tempSp --sigP_max $tempSp --sigB_min $tempSb --sigB_max $tempSb --fileName $Fname$cntTot --run_analysis true --run_Combine false --tau_ohmic $tau_ohmic --max_T_f $max_T_f --run_magnetars $run_magnetars --kill_dead $kill_dead &
        sleep 3
        cntTot=$(expr $cntTot + 1)
#        if (( $cntTot % ($SLURM_NTASKS-1) == 0 ))
#        then
#            echo "Waiting..." $cntTot
#            wait
#        fi
    done
done
wait


julia --threads 1 Gen_pop_fit.jl --NPts_Psig $NPts_Psig --NPts_Bsig $NPts_Bsig --Npts_B $Npts_B --Npts_P $Npts_P --Bmax $Bmax  --Bmin $Bmin --Pmax $Pmax --Pmin $Pmin --sigP_min $tempSp --sigP_max $tempSp --sigB_min $tempSb --sigB_max $tempSb --fileName $Fname --run_analysis false --run_Combine true --Nruns $cntTot --tau_ohmic $tau_ohmic --max_T_f $max_T_f --run_magnetars $run_magnetars --kill_dead $kill_dead


