import sys
import os 
from genesis4.lume_genesis_JT import Genesis4
from genesis4.lume_genesis_JT.match_fodo import match_to_FODO
from constants import *
import numpy as np

def scan_params(xlamds, sample, dgrid, ngrid, seed,                                                                 # Basics
                pulselen, sigma, peak_current, beamprofile = 'F', gam0 = np.around(3990./0.511,3),                   # Define e-beam
                 chirp = 20/0.511, delgam = 4/0.511, emitnx = 0.3e-6, emitny = 0.3e-6,                              # Define e-beam
                 xlamdu = 0.026, nwig = 130, fodo_length = 3.9*2, Kstart = 1.26, taper = 0,                                     # Define lattice
                 ustart = 2, ustop = 10, order = 2, quad_length = 0.084, quad_grad = 1.784478917,                            # Define lattice
                 phaseShifts = None, phaseshift_length = 0.0, linename = 'myline',                                    # Define lattice
                 seedfile = None, P0 = 0,                                                                            # Define Seed
                 workdir = None,
                 nametag = ""):

    #####################################################################################################
    ################################# Initialize Genesis4   #############################################
    #####################################################################################################
    
    # Template, note that insert and pop may need to be changed if template is changed.
    FILE = '$LCLS_LATTICE/genesis/version4/sc_hxr/genesis4_sc_hxr.in'                                 

    G = Genesis4(input_file = FILE, verbose = True, use_temp_dir = True)
 
    rootname = 'K' + str(np.round(Kstart,4)) + '_taper' + str(taper)  + '_' + nametag

    G.input['main'][0]['gamma0'] = gam0
    G.input['main'][0]['lambda0'] = xlamds
    
    G.input["main"][0]["rootname"] = rootname
    G.input['main'][0]['seed'] = seed

    G.input['main'][1]['slen'] = pulselen*CSPEED
    G.input['main'][1]['sample'] = sample


    G.input['main'][-1]['zstop'] = fodo_length*ustop/2
    
    # Add writing a beam (particle) file
    G.input['main'].append({'type':'write', 'beam':rootname})
    # Add writing a field file
    G.input['main'].append({'type':'write', 'field':rootname})

    #####################################################################################################
    ################################# Define Seed File   #############################################
    #####################################################################################################

    G.input['main'][2]['power'] = P0
    G.input['main'][2]['dgrid'] = dgrid
    G.input['main'][2]['ngrid'] = ngrid
    if seedfile:
        G.input['main'].pop(2)
        dic_import_field = {'type': 'importfield',
                             'file': seedfile,
                             'harmonic': 1}
        G.input['main'].insert(2, dic_import_field)


    #####################################################################################################
    ################################# Define initial e-beam #############################################
    #####################################################################################################
    
    # ----------------------------------make beam-------------------------------------------------------
    
    assert beamprofile == 'G' or beamprofile == 'F', "beamprofile must be either G (Gaussian) or F(flattop)"
    NPTS = 10000
    SLEN = pulselen*CSPEED
  

    S = np.linspace(0, SLEN, NPTS)
    GAMMA = np.linspace(gam0 - chirp/2, gam0 + chirp/2, NPTS)
    if beamprofile == 'F':
        CURRENT = np.ones((NPTS,))*peak_current
    else:
        assert sigma is not None, "sigma must be a number"
        sigma_z = sigma*CSPEED
        CURRENT = np.exp(-(S - np.mean(S))**2/2/sigma_z**2)
        CURRENT /= np.max(CURRENT)
        CURRENT *= peak_current
    
    # ----------------------------------configure in file--------------------------------------------------
    dic_current = {'type': 'profile_file',
       'label': rootname+'_beamcurrent',
       'xdata': S, 
       'ydata': CURRENT}
    dic_gamma = {'type': 'profile_file',
       'label': rootname + '_beamgamma',
       'xdata': S, 
       'ydata': GAMMA}

    dic_beam = {'type': 'beam',
                'current': '@' + rootname+'_beamcurrent',
                'gamma': '@' + rootname + '_beamgamma',
                'delgam': delgam,
                'ex': emitnx,
                'ey': emitny}

    #G.input['main'].pop(4)
    G.input['main'].pop(3)

    
    G.input['main'].insert(3, dic_current)
    G.input['main'].insert(4, dic_gamma)
    G.input['main'].insert(5, dic_beam)
   
    #####################################################################################################
    ################################# Define undulator lattice ##########################################
    #####################################################################################################

    G.make_new_lattice(undKs =[Kstart]*32, und_period = xlamdu, und_nperiods = nwig, fodo_length = fodo_length, 
                   quad_length = quad_length, quad_grad = quad_grad, 
                   phaseShifts = phaseShifts, phaseshift_length = phaseshift_length, 
                   latticefilepath = None, linename = linename, 
                   apply_taper = True, dKbyK = taper, ustart = ustart, ustop = ustop, order = order)

    L_drift = fodo_length/2 - quad_length
    betax, betay = match_to_FODO(gamma0 = gam0, emitnx = emitnx, emitny = emitny, L_quad=quad_length, L_drift = L_drift , kq = quad_grad)


    G.input['main'][5]['alphax'] = 0.0
    G.input['main'][5]['alphay'] = 0.0
    G.input['main'][5]['betax'] = betax
    G.input['main'][5]['betay'] = betay
    

    G.use_mpi = True

    G.batch_run()

    return G



######################################################################
archive_dir = "/sdf/data/ad/ard/u/jytang/test_genesis4/"

xlamds = 6.30524765e-11
sample = 10

#field
dgrid=100e-6
ngrid = 181
P0 = 2e8

#beam
pulselen = 20e-15
peak_current = 2e3
beamprofile = 'F'
sigma = None
gam0 = np.around(8000./0.511,3)
chirp = 0/0.511
delgam = 1/0.511
emitnx = 0.3e-6
emitny = 0.3e-6

#undualtor
Kstart = 0.4335
taper = 0
ustart = 2
ustop = 10
order = 2
quad_grad = 1.784478917



#############################################################################
Glist = []
G = scan_params(xlamds = xlamds, sample = sample,dgrid=dgrid, ngrid = ngrid, seed = np.random.randint(100000),          # Basics
                pulselen = pulselen, sigma = sigma, peak_current = peak_current, beamprofile = beamprofile, gam0 = gam0,                   # Define e-beam
                 chirp = chirp, delgam = delgam, emitnx = emitnx, emitny = emitny,                              # Define e-beam
                 xlamdu = 0.026, nwig = 130, fodo_length = 3.9*2, Kstart = Kstart, taper = taper,                                     # Define lattice
                 ustart = ustart, ustop = ustop, order = order, quad_length =  0.084, quad_grad = quad_grad,                            # Define lattice
                 phaseShifts = None, phaseshift_length = 0.0, linename = 'myline',                                    # Define lattice
                 seedfile = None, P0 = P0,                                                                            # Define Seed
                 nametag = "tt")
Glist.append(G)


if not os.path.exists(archive_dir):
    os.mkdir(archive_dir)


for G in Glist:
    G.wait_and_process(archive = archive_dir)


