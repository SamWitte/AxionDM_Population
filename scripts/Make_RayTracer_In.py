import numpy as np
import os
import glob

B0_L = np.logspace(11, np.log10(4.4e13), 10)
P_L = np.logspace(1e-2, np.log10(20), 10)
ThetaM = np.array([0.0])
mass_a = np.array([1.4e9]) * 2*np.pi * 6.58e-16 # eV
gagg = 1e-12 # 1/GeV
rNS = 13.0
M_NS = 1.4
vx_NS = 0.0
vy_NS = 0.0
vz_NS = 0.0
trajs=20000

eta_fill=0.2

path_out = "../"


num_scripts=2
arr_text = np.empty(num_scripts, dtype=object)

# write into to each script
# SERVER SPECIFIC -- writing for Hydra
for i in range(num_scripts):
    arr_text[i] = "#!/bin/bash -l \ncd ../src/RayTracing/ \nmodule load julia \n"
    arr_text[i] += "declare -i memPerjob \nmemPerjob=$((SLURM_MEM_PER_CPU/SLURM_NTASKS)) \n"
    arr_text[i] += "Trajs={:.0f} \nMassNS={:.2f} \nrNS={:.2f} \n".format(trajs, M_NS, rNS)
    arr_text[i] += "ftag=\"_\" \n"
    arr_text[i] += "vNS_x={:.2f} \nvNS_y={:.2f} \nvNS_z={:.2f} \n".format(vx_NS, vy_NS, vz_NS)
    arr_text[i] += "null_fill={:.2f}\n".format(eta_fill)

block_text = "for ((i = 0; i < $SLURM_NTASKS ; i++)); do \n\tsrun -n 1 --exclusive --mem=$memPerjob --cpus-per-task=1 julia --threads 1 Gen_Samples_GR_Server.jl --MassA $MassA --B0 $B0 --ThetaM $ThetaM --rotW $rotW--AxG $gagg --vNS_x $vNS_x --vNS_y $vNS_y --vNS_z $vNS_z --Mass_NS $MassNS --rNS $rNS  --Nts $Trajs --ftag $tagF$i --run_RT 1 --Flat false --Iso false --add_tau true --Mass_NS $MassNS --Thick true --side_runs $SLURM_NTASKS & \n\tsleep 3 \ndone \nwait \n srun --ntasks=1 julia --threads 1 Gen_Samples_GR_Server.jl --MassA $MassA --B0 $B0 --ThetaM $ThetaM --rotW $rotW --Nts $Trajs --run_RT 0 --run_Combine 1 --ftag $tagF --side_runs $SLURM_NTASKS --Mass_NS $MassNS --rNS $rNS --vNS_x $vNS_x --vNS_y $vNS_y --vNS_z $vNS_z \n\n"

indx = 0
for i1 in range(len(B0_L)):
    for i2 in range(len(P_L)):
        for i3 in range(len(ThetaM)):
            for i4 in range(len(mass_a)):
                newL = "B0={:.3e} \nrotW={:.4f} \nThetaM={:.4f} \nMassA={:.3e} \n".format(B0_L[i1], 2*np.pi / P_L[i2], ThetaM[i3], mass_a[i4])
                arr_text[indx%num_scripts] += newL + block_text
                indx += 1


for i in range(num_scripts):
    text_file = open("../temp_hold/Script_Run_RT_{:.0f}.sh".format(i), "w")
    text_file.write(arr_text[i])
    text_file.close()

    
