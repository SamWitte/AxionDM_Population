import numpy as np
import glob
import h5py
from scipy.special import erf, erfinv

def check_conversion(MassA, B, P, thetaM):
    ne_max = 2 * (2*np.pi / P) * (B * 1.95e-2) / 0.3 * 6.58e-16
    max_omegaP = np.sqrt(4*np.pi / 137 * ne_max / 5.11e5)
    if max_omegaP < MassA:
        return 0
    else:
        return 1

def Pulsar_signal(MassA, ThetaV, Bv, Pv, gagg=1.0e-12, eps_theta=0.03, dopplerS=0.0, density_rescale=0.45, v0_rescale=220.0):
    # Mass of axion
    # Viewing angle
    # Magnetic field of pulsar
    # Period of pulsar
    ##### Assumes rectangular grid!


    MassA_tag_temp = "Ma_{:.2e}".format(MassA)
    MassA_tag = MassA_tag_temp[:-2] + MassA_tag_temp[-1]
    Gagg_tag = "AxionG_{:.1e}".format(gagg)
    bankF = glob.glob("RayTracing/results/*"+MassA_tag+"*"+Gagg_tag+"*")
    
    # ASSUMING THETA_M = 0.0
    if check_conversion(MassA, Bv, Pv, 0.0) == 0:
        print("Not converting... ")
        return np.array([0.0, 0.0])
    
    file_params = []

    for i in range(len(bankF)):
        fN = bankF[i]
        
        ftag1 = "_B0_"
        ftag2 = "_P_"
        B0 = float(fN[fN.find(ftag1) + len(ftag1):fN.find(ftag2)])

        ftag1 = "_P_"
        ftag2 = "_ThetaM_"
        P = float(fN[fN.find(ftag1) + len(ftag1):fN.find(ftag2)])
        file_params.append([i, B0, P])

    file_params = np.asarray(file_params)
    Blist = np.sort(np.unique(file_params[:,1]))
    Plist = np.sort(np.unique(file_params[:,2]))

    argFlip_B = np.where(np.diff(np.sign(Blist - Bv)) != 0.0)[0]
    Bmin = Blist[argFlip_B]
    Bmax = Blist[argFlip_B + 1]
    weight_B = (np.log10(Bmax) - np.log10(Bv)) / (np.log10(Bmax) - np.log10(Bmin))

    argFlip_P = np.where(np.diff(np.sign(Plist - Pv)) != 0.0)[0]
    Pmin = Plist[argFlip_P]
    Pmax = Plist[argFlip_P + 1]
    weight_P = (Pmax - Pv) / (Pmax - Pmin)


    output = []

    # DEAL WITH POSSIBILITY THAT ONE FILE HAS ZEROS....
    # for moment i dont care
#    test_lowR = check_conversion(MassA, Bmin, Pmax, thetaM)
#    test_upR = check_conversion(MassA, Bmax, Pmax, thetaM)
#    test_lowL = check_conversion(MassA, Bmin, Pmin, thetaM)
#    if (lowR == 0)&&(test_upR == 0)&&(test_lowL == 0):
#        weight_P = 1.0
#        weight_B = 0.0
#    elif (lowR == 0)&&(test_upR == 0):
#
#    elif (lowR == 0)&&(test_lowL == 0):
#
#    elif (lowR == 0):
        
    
    
    # if check_conversion(MassA, Bmin, Pmax, thetaM) == 0:

    # find each file to interpolate
    file1 = np.all(np.column_stack((file_params[:, 1]==Bmin, file_params[:,2]==Pmin)), axis=1)
    fileL = h5py.File(bankF[int(file_params[file1, 0])], "r")
    weight = weight_P * weight_B
    
    ThetaVals = fileL["thetaX_final"][:]
    rates = fileL["weights"][:]
    ergs = fileL["erg_loc"][:] * fileL["red_factor"][:] # the second factor is only there because of a mistake before
    rates_small = theta_cut(rates, ThetaVals, ThetaV, eps=eps_theta)
    ergs_small = theta_cut(ergs, ThetaVals, ThetaV, eps=eps_theta)
    photons = rates_small / ( np.sin(ThetaV) * 2 * np.sin(eps_theta)) / Pv  # eV / s
    for jj in range(len(ergs_small)):
        output.append([ergs_small[jj], photons[jj] * weight])
   

    file1 = np.all(np.column_stack((file_params[:, 1]==Bmax, file_params[:,2]==Pmax)), axis=1)
    fileL = h5py.File(bankF[int(file_params[file1, 0])], "r")
    weight = (1 - weight_P) * (1 - weight_B)
    # get stuff

    ThetaVals = fileL["thetaX_final"][:]
    rates = fileL["weights"][:]
    ergs = fileL["erg_loc"][:]
    rates_small = theta_cut(rates, ThetaVals, ThetaV, eps=eps_theta)
    ergs_small = theta_cut(ergs, ThetaVals, ThetaV, eps=eps_theta)
    photons = rates_small / ( np.sin(ThetaV) * 2 * np.sin(eps_theta)) / Pv  # eV / s
    for jj in range(len(ergs_small)):
        output.append([ergs_small[jj], photons[jj] * weight])


    
    file1 = np.all(np.column_stack((file_params[:, 1]==Bmin, file_params[:,2]==Pmax)), axis=1)
    fileL = h5py.File(bankF[int(file_params[file1, 0])], "r")
    weight = (1 - weight_P) * weight_B
    # get stuff

    ThetaVals = fileL["thetaX_final"][:]
    rates = fileL["weights"][:]
    ergs = fileL["erg_loc"][:]
    rates_small = theta_cut(rates, ThetaVals, ThetaV, eps=eps_theta)
    ergs_small = theta_cut(ergs, ThetaVals, ThetaV, eps=eps_theta)
    photons = rates_small / ( np.sin(ThetaV) * 2 * np.sin(eps_theta)) / Pv  # eV / s
    for jj in range(len(ergs_small)):
        output.append([ergs_small[jj], photons[jj] * weight])
    
    file1 = np.all(np.column_stack((file_params[:, 1]==Bmax, file_params[:,2]==Pmin)), axis=1)
    fileL = h5py.File(bankF[int(file_params[file1, 0])], "r")
    weight = weight_P * (1 - weight_B)
    # get stuff

    ThetaVals = fileL["thetaX_final"][:]
    rates = fileL["weights"][:]
    ergs = fileL["erg_loc"][:]
    rates_small = theta_cut(rates, ThetaVals, ThetaV, eps=eps_theta)
    ergs_small = theta_cut(ergs, ThetaVals, ThetaV, eps=eps_theta)
    photons = rates_small / ( np.sin(ThetaV) * 2 * np.sin(eps_theta)) / Pv  # eV / s
    for jj in range(len(ergs_small)):
        output.append([ergs_small[jj], photons[jj] * weight])
    
    
    output = np.asarray(output)
    #
    erg_preD = MassA
    vel_samp = np.sum(v0_rescale * erfinv(np.random.rand(len(output[:,0]), 3)), axis=1) / 2.998e5
    erg_preD *= (1+vel_samp**2)
    output[:,0] = erg_preD * (1 + dopplerS)
    output[:, 1] *= (density_rescale / 0.45)
    output[:, 1] *= (220.0 / v0_rescale)
    # add v0_rescale
    
    return output



def theta_cut(arrayVals, Theta, thetaV, eps=0.01):
    if (thetaV - eps) < 0:
        condition1 = Theta < (thetaV + eps)
        condition2 = Theta > np.pi + (thetaV - eps)
        jointC = condition1
    elif (thetaV - eps) > np.pi:
        condition1 = Theta < (thetaV + eps) - np.pi
        condition2 = Theta > (thetaV - eps)
        jointC = Theta > (thetaV - eps)
    else:
        condition1 = Theta < (thetaV + eps)
        condition2 = Theta > (thetaV - eps)
    
        jointC = np.all(np.column_stack((condition1, condition2)), axis=1)
    return arrayVals[jointC]

