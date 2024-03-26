#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-

# rfp - radiation field propagator
# tools for manipulating genesis dfl field files
# dfl file assuming Genesis2 indexing dfl[t, x, y]
# J. Duris jduris@slac.stanford.edu
# Modified by J. Tang jytang@stanford.edu.

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from constants import *
    
def fwhm(array, nkeep=2, plotQ=False, relcut=0.5, abscut=None):
    array_max = array.max()
    arg_max = array.argmax()
    if type(abscut) is type(None):
        scaled = array-relcut*array_max
    else:
        scaled = array-abscut
    inds = np.array(range(len(array)))
    
    # find lower crossing
    try:
        xlow = np.max(inds[(inds < arg_max) * (scaled < 0.)])
        xlows = xlow + np.array(range(nkeep)) - (nkeep-1)/2
        xlows = xlows[xlows > 0]
        xlows = np.array(xlows, dtype=np.int)
        if len(xlows):
            pfl = np.polyfit(xlows,scaled[xlows],1)
            xlow = -pfl[1]/pfl[0]
        else:
            xlow = np.nan
    except:
        xlow = np.nan
    
    # find upper crossing
    try:
        xhigh = np.min(inds[(inds > arg_max) * (scaled < 0.)])
        xhighs = xhigh + np.array(range(nkeep)) - (nkeep-1)/2 - 1
        xhighs = xhighs[xhighs > 0]
        xhighs = np.array(xhighs, dtype=np.int)
        if len(xhighs):
            pfh = np.polyfit(xhighs,scaled[xhighs],1)
            xhigh = -pfh[1]/pfh[0]
        else:
            xhigh = np.nan
    except:
        xhigh = np.nan
                
    if plotQ:
        import matplotlib.pyplot as plt
        try:
            plt.plot(xlows,scaled[xlows],'o')
            plt.plot(xlow,0,'o')
            plt.show()
        except:
            pass
        try:
            plt.plot(xhighs,scaled[xhighs],'o')
            plt.plot(xhigh,0,'o')
            plt.show()
        except:
            pass
        
    # determine width
    return np.array([np.abs(xhigh-xlow),np.abs(arg_max-xlow),np.abs(arg_max-xhigh),xlow,xhigh])


# maximum width at half max
def mwhm(array, nkeep=2, plotQ=False, relcut=0.5, abscut=None):
    array_max = array.max()
    arg_max = array.argmax()
    if type(abscut) is type(None):
        scaled = array-relcut*array_max
    else:
        scaled = array-abscut
    inds = np.array(range(len(array)))
    
    # find lower crossing
    try:
        xlow = np.min(inds[(inds < arg_max) * (scaled > 0.)])
        xlows = xlow + np.array(range(nkeep)) - (nkeep-1)/2
        xlows = xlows[xlows > 0]
        xlows = np.array(xlows, dtype=np.int)
        if len(xlows):
            pfl = np.polyfit(xlows,scaled[xlows],1)
            xlow = -pfl[1]/pfl[0]
        else:
            xlow = np.nan
    except:
        xlow = np.nan
    
    # find upper crossing
    try:
        xhigh = np.max(inds[(inds > arg_max) * (scaled > 0.)])
        xhighs = xhigh + np.array(range(nkeep)) - (nkeep-1)/2 - 1
        xhighs = xhighs[xhighs > 0]
        xhighs = np.array(xhighs, dtype=np.int)
        if len(xhighs):
            pfh = np.polyfit(xhighs,scaled[xhighs],1)
            xhigh = -pfh[1]/pfh[0]
        else:
            xhigh = np.nan
    except:
        xhigh = np.nan
                
    if plotQ:
        try:
            plt.plot(xlows,scaled[xlows],'o')
            plt.plot(xlow,0,'o')
            plt.show()
        except:
            pass
        try:
            plt.plot(xhighs,scaled[xhighs],'o')
            plt.plot(xhigh,0,'o')
            plt.show()
        except:
            pass
        
    # determine width
    return np.array([np.abs(xhigh-xlow),np.abs(arg_max-xlow),np.abs(arg_max-xhigh),xlow,xhigh])


def make_gaus_slice(ncar=251, dgrid=400.e-6, w0=40.e-6):
    
    xs = np.linspace(-1,1,ncar) * dgrid
    ys = xs
    xv, yv = np.meshgrid(xs, ys) 
    
    sigx2 = (w0 / 2.)**2;
    fld = np.exp( -0.25 * (xv**2 + yv**2) / sigx2 ) + 1j * 0
    fld = fld[:,:, None]
    
    return fld

def make_gaus_beam(ncar=251, dgrid=400.e-6, w0=40.e-6, dt=1e-6/3e8, t0=0., nslice=11, trms=2e-6/3e8):

    
    xs = np.linspace(-1,1,ncar) * dgrid
    ys = xs
    xv, yv = np.meshgrid(xs, ys) 
    
    sigx2 = (w0 / 2.)**2;
    fld = np.exp( -0.25 * (xv**2 + yv**2) / sigx2 )
    
    ts = dt * np.arange(nslice); ts -= np.mean(ts)
    amps = np.exp(-0.25 * ((ts-t0)/trms)**2)
    
    fld0 = np.zeros([nslice, ncar, ncar]) + 1j * 0.
    
    for ia, a in enumerate(amps):
        fld0[ia] = a * fld
    
    return fld0


def pad_dfl(fld, pads):
    # pads should be of the form [[i0lo,i0hi],[i1lo,i1hi],[i2lo,i2hi]]
    fld = np.pad(fld, pads)
    return fld
    
def pad_dfl_t(fld, pads):
    fld = np.pad(fld, [pads,[0,0],[0,0]])
    return fld
    
def pad_dfl_x(fld, pads):
    fld = np.pad(fld, [[0,0],pads,[0,0]])
    return fld

def pad_dfl_slice_x(fld_slice, pads):
    fld_slice = np.pad(fld_slice, [pads,[0,0]])
    return fld_slice

def pad_dfl_slice_xy(fld_slice, pads): 
    fld_slice = np.pad(fld_slice, [pads,pads])
    return fld_slice


def pad_dfl_y(fld, pads):
    fld = np.pad(fld, [[0,0],[0,0],pads])
    return fld
    
def pad_dfl_xy(fld, pads):
    fld = np.pad(fld, [[0,0],pads,pads])
    return fld
    
def unpad_dfl(fld, pads):
    fld = fld[pads[0,0]:-pads[0,1],pads[1,0]:-pads[1,1],pads[2,0]:-pads[2,1]]
    return fld
    
def unpad_dfl_t(fld, pads):
    fld = fld[pads[0]:-pads[1],:,:]
    return fld
    
def unpad_dfl_x(fld, pads):
    fld = fld[:,pads[0]:-pads[1],:]
    return fld
    
def unpad_dfl_y(fld, pads):
    fld = fld[:,:,pads[0]:-pads[1]]
    return fld

def unpad_dfl_slice_x(fld_slice, pads):
    fld_slice = fld_slice[pads[0]:-pads[1],:]
    return fld_slice

def unpad_dfl_slice_xy(fld_slice, pads):
    fld_slice = fld_slice[pads[0]:-pads[1],pads[0]:-pads[1]]
    return fld_slice

def unpad_dfl_xy(fld, pads):
    fld = fld[:,pads[0]:-pads[1],pads[0]:-pads[1]]
    return fld

# slip field forward along the temporal grid and slip in zeros in the tail
def slip_fld(fld, dt, slippage_time, trailing_scale_factor=1e-6):
    nslip = int(np.floor(slippage_time / dt))
    fld2 = np.roll(fld,nslip,axis=0)
    fld2[:nslip] *= trailing_scale_factor
    # maybe should advance the phase of the radiation by the remainder slippage time also...
    return fld2
    
# select only slices with significant power
def threshold_slice_selection(fld, slice_processing_relative_power_threshold = 1e-6, verboseQ=False):
        #slice_processing_relative_power_threshold = 1e-6 # only propagate slices where there's beam (0 disables)
        pows = np.sum(np.abs(fld)**2,axis=(1,2))
        slice_selection = pows >= np.max(pows) * slice_processing_relative_power_threshold
        #if verboseQ:
        u0 = np.sum(pows); u1 = np.sum(pows[slice_selection])
        print('INFO: threshold_slice_selection - Fraction of power lost is',1.-u1/u0,'for slice_processing_relative_power_threshold of',slice_processing_relative_power_threshold)
        return slice_selection

# Siegman collimating transform
# NOTE: dx should be 2*dgrid/ncar
def st2(fld, lambda_radiation, dgridin, dgridout, ABDlist, ncar, verboseQ = False, outQ = False):

   tau = 6.283185307179586
   scale=1.; M=dgridout/dgridin

   if outQ:
      dx = 2.*dgridout/(ncar-1.)
      phasefactor = (1./M-ABDlist[2])*dx*dx*tau/2./lambda_radiation/ABDlist[1];
      scale = dgridout; #for genesis, each cell is intensity so don't need this
      if verboseQ:
          print("(1./M-ABDlist[2]) = ",(1./M-ABDlist[2]), "\tdx*dx*tau/2. = ", dx*dx*tau/2., "\tlambda = " , lambda_radiation , "\tABDlist[1] = " , ABDlist[1]);

   else:
      dx = 2.*dgridin/(ncar-1.);
      phasefactor = (M-ABDlist[0])*dx*dx*tau/2./lambda_radiation/ABDlist[1];
      scale = dgridin; #//for genesis, each cell is intensity so don't need this
      if verboseQ:
          print("(M-ABDlist[0]) = ", (M-ABDlist[0]), "\tdx*dx*tau/2. = ", dx*dx*tau/2., "\tlambda = ", lambda_radiation, "\tABDlist[1] = ", ABDlist[1])

   dxscale = dx / scale;
   
   # calculate the phase mask
   igrid = np.arange(ncar)-np.floor(ncar/2)
   phases = phasefactor * (igrid ** 2.)
   pxv, pyv = np.meshgrid(phases, phases)
   phasor_mask = np.exp(1j * (pxv + pyv))
   
   # apply the phase mask to each slice
   fld *= phasor_mask
   
   return fld

# inverse Siegman collimating transform
def ist2(fld, lambda_radiation, dgridin, dgridout, ABDlist, ncar, verboseQ = False):
   return st2(fld, lambda_radiation, dgridin, dgridout, ABDlist, ncar, verboseQ, True)

# Siegman collimated Huygen's kernel
def sk2(fld, lambda_radiation, dgridin, dgridout, ABDlist, ncar, verboseQ = False):
#/* Propagate radiation stored in data. */

   tau = 6.283185307179586;
   M=dgridout/dgridin;
   
   #//Nc = M*pow(dgridin,2.)/lambda_radiation/ABDlist[1]; # collimated Fresnel number 
   Nc = M*np.power(2.*dgridin,2.)/lambda_radiation/ABDlist[1]; # collimated Fresnel number 
   phasefactor = tau/2./Nc;
   
   # calculate the phase mask
   midpoint = np.floor(ncar/2)
   igrid = np.fft.ifftshift(np.arange(ncar)-midpoint)
   phases = phasefactor * igrid ** 2.
   pxv, pyv = np.meshgrid(phases, phases)
   phasor_mask = np.exp(1j * (pxv + pyv))
   
   # apply the phase mask to each slice
   fld *= phasor_mask
   
   return fld
   

def fld_info(fld, dgrid = 400.e-6, dt=1e-6/3e8,verbose = False):
    
    power = np.abs(fld)**2
        
    nslice = fld.shape[0]
   
    ncar = fld.shape[1]
    norm = np.sum(power)
    tproj = np.sum(power, axis=(1,2))
    xproj = np.sum(power, axis=(0,2))
    yproj = np.sum(power, axis=(0,1))
    energy_uJ = norm*dt * 1e6
    
    transverse_grid = np.linspace(-1,1,ncar) * dgrid * 1e6
    ts = dt * np.arange(nslice); ts -= np.mean(ts); ts= ts * 1e15
    xs = transverse_grid
    ys = transverse_grid
    
    tmean = np.dot(ts, tproj)/norm
    xmean = np.dot(xs, xproj) / norm
    ymean = np.dot(ys, yproj) / norm
    trms = np.sqrt(np.dot(ts**2, tproj) / norm - tmean**2)
    xrms = np.sqrt(np.dot(xs**2, xproj) / norm - xmean**2)
    yrms = np.sqrt(np.dot(ys**2, yproj) / norm - ymean**2)
    dx = xs[1] - xs[0]; dy = ys[1] - ys[0]; dt = ts[1] - ts[0]
    xfwhm = fwhm(xproj) * dx
    yfwhm = fwhm(yproj) * dy
    tfwhm = fwhm(tproj) * dt
    
    maxpower = np.amax(tproj)/1e9
    
    ndecimals = 12
    xmean = np.around(xmean, ndecimals); ymean = np.around(ymean, ndecimals);tmean = np.around(tmean, ndecimals)
    xrms = np.around(xrms, ndecimals); yrms = np.around(yrms, ndecimals); trms = np.around(trms, ndecimals)
    xfwhm = np.around(xfwhm, ndecimals)[0]; yfwhm = np.around(yfwhm, ndecimals)[0];  tfwhm = np.around(tfwhm, ndecimals)[0]
    energy_uJ = np.around(energy_uJ, ndecimals)
    maxpower = np.around(maxpower, ndecimals)
    #print('norm =',norm,'   x,y mean =',xmean,ymean, '   x,y rms =', xrms,yrms, '   wx,wy =', 2*xrms,2*yrms)

    
    
    #xmean *= 1e6; ymean *= 1e6; xrms *= 1e6; yrms *= 1e6;
    if verbose:
        print('energy = ' + str(energy_uJ) + 'uJ, ' + 'peakpower = ' + str(maxpower) + 'GW, '
         +'trms = '+str(trms) + 'fs, ' + 'tfwhm = ' + str(tfwhm) + 'fs, '
         +'xrms = '+str(xrms) + 'um, ' + 'xfwhm = ' + str(xfwhm) + 'um, '
        +'yrms = '+str(xrms) + 'um, ' + 'yfwhm = ' + str(yfwhm) + 'um, ')
    
    
    return energy_uJ, maxpower, tmean, trms, tfwhm, xmean, xrms, xfwhm, ymean, yrms, yfwhm


def fld_slice_info(fld, dgrid = 400.e-6):
    
    power = np.abs(fld)**2
        
    nslice = fld.shape[0]
   
    ncar = fld.shape[1]
    norm = np.sum(power)
    
    xproj = np.sum(power, axis=(0,2))
    yproj = np.sum(power, axis=(0,1))
    
    
    transverse_grid = np.linspace(-1,1,ncar) * dgrid * 1e6
    xs = transverse_grid
    ys = transverse_grid
    
    xmean = np.dot(xs, xproj) / norm
    ymean = np.dot(ys, yproj) / norm
   
    xrms = np.sqrt(np.dot(xs**2, xproj) / norm - xmean**2)
    yrms = np.sqrt(np.dot(ys**2, yproj) / norm - ymean**2)
    dx = xs[1] - xs[0]; dy = ys[1] - ys[0];
    xfwhm = fwhm(xproj) * dx
    yfwhm = fwhm(yproj) * dy
   
    
   
    
    ndecimals = 6
    xmean = np.around(xmean, ndecimals); ymean = np.around(ymean, ndecimals);
    xrms = np.around(xrms, ndecimals); yrms = np.around(yrms, ndecimals); 
    xfwhm = np.around(xfwhm, ndecimals)[0]; yfwhm = np.around(yfwhm, ndecimals)[0];  
   
    #print('norm =',norm,'   x,y mean =',xmean,ymean, '   x,y rms =', xrms,yrms, '   wx,wy =', 2*xrms,2*yrms)

    
    
    #xmean *= 1e6; ymean *= 1e6; xrms *= 1e6; yrms *= 1e6;
    print('xrms = '+str(xrms) + 'um, ' + 'xfwhm = ' + str(xfwhm) + 'um, '
        +'yrms = '+str(xrms) + 'um, ' + 'yfwhm = ' + str(yfwhm) + 'um, ')
    
    
    return xrms, xfwhm, yrms, yfwhm


def get_spectrum(dfl, zsep, xlamds, npad = (0,0), onaxis=True):
    h_Plank = H_PLANK
    c_speed  = CSPEED
    dt = zsep*xlamds/c_speed
    
    nslice = dfl.shape[0]
    ncar = dfl.shape[1]
    s = np.arange(nslice + npad[0] + npad[1]) * dt
    s_fs = s*1e15
    
    
    ws = np.arange(s_fs.shape[0]) / (s_fs[-1] - s_fs[0])
    ws -= np.mean(ws)
    hw0 = h_Plank * c_speed /xlamds
    hws = h_Plank*1e15 * ws + hw0
    
    if onaxis:
        field = dfl[:,ncar//2 +1, ncar//2 + 1]
        field = np.pad(field, npad)
        ftfld = np.fft.fftshift(np.fft.fft(field))
        spectra = np.abs(ftfld)**2
        #spectra /= np.sum(spectra)
    
    else:
        field = np.pad(dfl, (npad, (0, 0), (0, 0)))
        ftfld = np.fft.fftshift(np.fft.fft(field, axis = 0), axes = 0)
        spectra = np.sum(np.abs(ftfld)**2, axis = (1, 2))
    
    return hws, spectra
        
