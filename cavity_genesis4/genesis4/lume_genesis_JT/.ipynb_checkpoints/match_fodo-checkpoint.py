
## Author: J. Tang, J. Duris
## Date: 3/25/2024
import numpy as np
from constants import *

def match_to_FODO(gamma0, emitnx, emitny, L_quad=10*0.026, L_drift=150*0.026, kq=14.584615, verbose = True):

    """
    return  matched twiss for a FODO lattice. The lattice is designed to start with half FODO, so alpha(z = 0) = 0
    L_quad: quad length
    L_drift: length between quads
    
    """

    c0 = CSPEED
    mc2 = gamma0*MC2
    
    xemit = emitnx/gamma0
    yemit = emitny/gamma0
    
    # calculate matched beta functions

    Lq = L_quad # quad length
    Ld = L_drift # undulator section length
    g = kq*mc2/c0 # quad gradient
   
    f = mc2/g*Lq
    #kq = g*c0/mc2
    quadP = np.sqrt(kq)*Lq/2.

    MF = [[np.cos(quadP), np.sin(quadP)/np.sqrt(kq)], [-np.sqrt(kq)*np.sin(quadP), np.cos(quadP)]]
    MD = [[np.cosh(quadP), np.sinh(quadP)/np.sqrt(kq)], [np.sqrt(kq)*np.sinh(quadP), np.cosh(quadP)]]
    ML = [[1, Ld], [0, 1]]

    A = np.dot(MF,np.dot(ML,np.dot(MD,np.dot(MD,np.dot(ML,MF))))) #MF*ML*MD*MD*ML*MF;
    Cphi = A[0,0]
    betaMAX = A[0,1]/np.sqrt(1.-Cphi**2)

    B = np.dot(MD,np.dot(ML,np.dot(MF,np.dot(MF,np.dot(ML,MD))))) #MD*ML*MF*MF*ML*MD;
    Cphi = B[0,0]
    betaMIN = B[0,1]/np.sqrt(1.-Cphi**2)

    xrms_match = np.sqrt(betaMAX * xemit)
    yrms_match = np.sqrt(betaMIN * yemit)
    xprms_match = xemit / xrms_match
    yprms_match = yemit / yrms_match

    if verbose:
        print('beta_x = ', betaMAX)
        print('beta_y = ', betaMIN)
        print('xrms_match = ', xrms_match)
        print('yrms_match = ', yrms_match)
        print('xprms_match = ', xprms_match)
        print('yprms_match = ', yprms_match)
        print('xemit = ', xemit)
        print('yemit = ', yemit)

    return betaMAX, betaMIN