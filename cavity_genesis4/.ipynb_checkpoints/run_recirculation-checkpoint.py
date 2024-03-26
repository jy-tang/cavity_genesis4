import numpy as np
import matplotlib.pyplot as plt
import time, os, sys
from run_genesis4 import *
import subprocess
sys.path.append("/sdf/group/ad/beamphysics/jytang/lume-genesis/") 

############################################################
workdir = '/sdf/scratch/users/j/jytang'
save_dir = "/sdf/data/ad/ard/u/jytang/test_genesis4/"

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

##################################################################


run_genesis4(xlamds = xlamds, sample = sample,dgrid=dgrid, ngrid = ngrid, seed = np.random.randint(100000),          # Basics
                pulselen = pulselen, sigma = sigma, peak_current = peak_current, beamprofile = beamprofile, gam0 = gam0,                   # Define e-beam
                 chirp = chirp, delgam = delgam, emitnx = emitnx, emitny = emitny,                              # Define e-beam
                 xlamdu = 0.026, nwig = 130, fodo_length = 3.9*2, Kstart = Kstart, taper = taper,                                     # Define lattice
                 ustart = ustart, ustop = ustop, order = order, quad_length = 0.084, quad_grad = quad_grad,                            # Define lattice
                 phaseShifts = None, phaseshift_length = 0.0, linename = 'myline',                                    # Define lattice
                 seedfile = None, P0 = 0,                                                                            # Define Seed
                 workdir = workdir, save_dir = save_dir, 
                 nametag = "")