import os

def file_outName(output_dir, M_a, tag, idxN, tau_ohmic, B0, P0, sB, sP, return_pop=False):
    
    fileO = tag + "TauO_{:.2e}_B_{:.2f}_sB_{:.2f}_P_{:.2f}_sP_{:.2f}_".format(tau_ohmic, B0, sB, P0, sP)

    
    fileO += "/"
    if not os.path.exists(output_dir + fileO):
        os.mkdir(output_dir + fileO)
        
    if return_pop:
        fileO += "Ma_{:.3e}/".format(M_a)
        if not os.path.exists(output_dir + fileO):
            os.mkdir(output_dir + fileO)
        fileO += "Pop_{:.0f}/".format(idxN)
        if not os.path.exists(output_dir + fileO):
            os.mkdir(output_dir + fileO)
    else:
        fileO += "Population_Details.txt"
    return output_dir + fileO
