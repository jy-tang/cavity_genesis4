import numpy as np
import matplotlib.pyplot as plt
import time, os, sys
from genesis4.run_genesis4 import *
import subprocess
from constants import *
from genesis4.lume_genesis_JT.readers import load_genesis4_fields
from genesis4.lume_genesis_JT.writers import write_dfl_genesis4_h5
from recirculation.run_mergeFiles_sdf import start_mergeFiles
from recirculation.run_recirculation_sdf import start_recirculation_newconfig
import gc
import h5py
#############################################################
root_dir = "/sdf/data/ad/ard/u/jytang/cavity_genesis4_20keV/"
folder_name = 'test1'
save_dir = root_dir + '/' + folder_name
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
############################################################
#Genesis

xlamds = 6.30524765e-11
sample = zsep = 100
dt = xlamds*sample/CSPEED


#field
dgrid=100e-6
ncar = 181
P0 = 0

#beam
pulselen = 20e-15
nslice = int(pulselen/dt)
print('number of slices =  ', nslice )

peak_current = 2e3
beamprofile = 'F'
sigma = None
gam0 = np.around(8000./0.511,3)
beam_chirp = 0/0.511
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

##################################################################
#Recirculation
nRoundtrips = 0        # number of iteration between ebeam shots
nEbeam_flat_init = 9   # number of ebeam shots
nEbeam_flat = 5
nEbeam_chirp = 2

isradi = 1
npadt = (8192 - nslice//isradi)//2
assert npadt >= 0, 'Error: npadt < 0 !'

npad1 = (512-ncar)//2
npadx = [int(npad1), int(npad1) + 1]

###################################################################
#Initialization
if nEbeam_flat_init > 0 and nEbeam_flat > 0:
    Nshot_total = nEbeam_flat_init + nEbeam_flat*nEbeam_chirp + 1
else:
    Nshot_total = nEbeam_chirp + 1

Nstart = 0
Nend = 2
##################################################################


for k in range(Nstart, Nend):
    nametag = 'n' + str(k)
    with open(save_dir + '/'+nametag+'_recirc.txt', "w") as myfile:
        myfile.write("Round energy/uJ peakpower/GW tmean/fs trms/fs  tfwhm/fs xmean/um xrms/um  xfwhm/um xmean/um  yrms/um yfwhm/um \n")
    with open(save_dir + '/'+nametag+'_transmit.txt', "w") as myfile:
        myfile.write("Round energy/uJ peakpower/GW tmean/fs trms/fs  tfwhm/fs  xmean/um xrms/um  xfwhm/um ymean/um yrms/um yfwhm/um \n")


#---------------------Prepare Seed Beam ------------------------------

    if k == 0:   # the first shot starts from noise
        dfl_filename = None
        seed_filename = None
    else:        # others start from recirculated seed
        dfl_filename = nametag+'_seed_init.fld.h5'  # the seed file without cutting from the last electron shots
        seed_filename =  nametag+'_seed.fld.h5'

        # read seed file
        print('start to cut seed file')
        readfilename = save_dir +'/'+dfl_filename
        with h5py.File(readfilename, 'r') as h5:
            fld, param = load_genesis4_fields(h5)
        print(fld.shape)
        print("finished read dfl")
        
        # cut the central part to match the size of shaped ebeam
        # get the max power position
        power = np.sum(np.abs(fld)**2, axis = (0,1))
        c = np.argmax(power)
        shift = int(len(power)//2 - c)
        fld = np.roll(fld, shift, axis = 2)
        
        power = np.sum(np.abs(fld)**2, axis = (0,1))
        c = np.argmax(power)
        fld = fld[:, :, int(c - nslice/2):int(c + nslice/2)]
        param['slicecount'] = fld.shape[2]
        print('fld shape after cutting', fld.shape)
        #write_dfl(fld, filename = root_dir+'/' + folder_name + '/'+seed_filename)
        write_dfl_genesis4_h5(fld = fld, param = param, filename = save_dir + '/' + seed_filename, indexing = 'Genesis4')

        del fld

    #-------------------change beam chirp-----------------------------------------------------
    if nEbeam_flat_init > 0 and  nEbeam_flat > 0 and (k < nEbeam_flat_init or (k - nEbeam_flat_init) %nEbeam_flat != 0):
        print('shot ', str(k), 'flat beam')
        chirp = 0
    else:
        chirp = beam_chirp

    #-----------------Start simulation ----------------------------------------------------------
    #Genesis
    t0 = time.time()
    sim_name = run_genesis4(xlamds = xlamds, sample = sample,dgrid=dgrid, ngrid = ncar, seed = np.random.randint(100000),          # Basics
                pulselen = pulselen, sigma = sigma, peak_current = peak_current, beamprofile = beamprofile, gam0 = gam0,                   # Define e-beam
                 chirp = chirp, delgam = delgam, emitnx = emitnx, emitny = emitny,                              # Define e-beam
                 xlamdu = 0.026, nwig = 130, fodo_length = 3.9*2, Kstart = Kstart, taper = taper,                                     # Define lattice
                 ustart = ustart, ustop = ustop, order = order, quad_length =  0.084, quad_grad = quad_grad,                            # Define lattice
                 phaseShifts = None, phaseshift_length = 0.0, linename = 'myline',                                    # Define lattice
                 seedfile = seed_filename, P0 = P0,                                                                            # Define Seed
                 save_dir = save_dir, 
                 nametag = nametag)

    print('It takes ', time.time() - t0, ' seconds to finish Genesis. Start recirculation')
        
    #recirculation
    t0 = time.time()
    jobid = start_recirculation_newconfig(zsep = zsep, ncar = ncar, dgrid = dgrid, nslice = nslice, xlamds=xlamds,           # dfl params
                                 npadt = npadt, Dpadt = 0, npadx = npadx,isradi = isradi,    # padding params
                                 l_undulator = 32*3.9, l_cavity = 149, w_cavity = 1, d1 = 100e-6, d2 = 100e-6, # cavity params
                                  verboseQ = 1, # verbose params
                                 nRoundtrips = nRoundtrips,               # recirculation params
                                 readfilename = sim_name + '.fld.h5' , 
                                 seedfilename = 'n' + str(k + 1)+'_seed_init.fld.h5',
                                       workdir = save_dir + '/' , saveFilenamePrefix = nametag)
    print('It takes ', time.time() - t0, ' seconds to finish recirculation.')

    # merge files for each roundtrip on nRoundtrips workers, with larger memory
    t0 = time.time()
    jobid = start_mergeFiles(nRoundtrips =nRoundtrips, workdir = save_dir + '/', saveFilenamePrefix=nametag, dgrid = dgrid, dt = dt, Dpadt = 0)
    print('It takes ', time.time() - t0, ' seconds to finish merging files.')
    gc.collect()
    