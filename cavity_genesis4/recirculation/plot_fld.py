### J. Duris, modified by J. Tang
## fld array assuming Genesis 4 indexing order fld[x, y, t]
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import time
import os 

# change default plotting font size
import matplotlib
font = {'family' : 'normal', 'size' : 14} # reasonable for on-screen displays
font = {'family' : 'normal', 'size' : 22} # for smaller plots in figures
#font['weight'] = 'bold'
matplotlib.rc('font', **font)

# jetvar: jet color map with white min
# https://stackoverflow.com/questions/9893440/python-matplotlib-colormap
cdict = {'red': ((0., 1, 1), (0.05, 1, 1), (0.11, 0, 0), (0.66, 1, 1), (0.89, 1, 1), (1, 0.5, 0.5)),
         'green': ((0., 1, 1), (0.05, 1, 1), (0.11, 0, 0), (0.375, 1, 1), (0.64, 1, 1), (0.91, 0, 0), (1, 0, 0)),
         'blue': ((0., 1, 1), (0.05, 1, 1), (0.11, 1, 1), (0.34, 1, 1), (0.65, 0, 0), (1, 0, 0))}
jetvar_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict,256)
cmap = jetvar_cmap
# cmap = 'jet'
#cmap = 'viridis'

try:
    # inferno reversed colormap with white background
    from matplotlib import cm
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    inferno_r_cmap = cm.get_cmap('inferno_r')
    # my_cmap.set_under('w') # don't seem to work
    xr = np.linspace(0, 1, 256)
    inferno_r_cmap_listed = inferno_r_cmap(xr)
    inferno_r_whitebg_cmap_listed = np.vstack((np.array([np.ones(4)+(inferno_r_cmap_listed[0]-np.ones(4))*x for x in np.linspace(0,1,int(256/8))]),inferno_r_cmap_listed[:-int(256/16)]))
    inferno_r_whitebg_cmap = ListedColormap(inferno_r_whitebg_cmap_listed)
    cmap = inferno_r_whitebg_cmap
except:
    pass

# note: if slice < 0, then this thing marginalizes axis -slice == [1,2,3]
def plot_fld_slice(fld, dgrid = 400.e-6, dt=1e-6/3e8, slice=None, ax=None, saveFilename=None, showPlotQ=True, savePlotQ=True, plotPowerQ=False, logScaleQ=False):
    ########## Switch axes due to the different definition between Genesis 2 and Genesis 4
    fld = np.moveaxis(fld, -1, 0)
    ###########################################################################################
    try:
        if slice == None:
            power = np.abs(fld[0])**2
        elif slice == -1:
            power = np.sum(np.abs(fld)**2, axis=1)
        elif slice == -2:
            power = np.sum(np.abs(fld)**2, axis=2)
        elif slice == -3:
            power = np.sum(np.abs(fld)**2, axis=0)
        else:
            power = np.abs(fld[slice])**2
        nslice = fld.shape[0]
    except:
        power = np.abs(fld)**2
        nslice = 1
    ncar = fld.shape[1]
    norm = np.sum(power)
    xproj = np.sum(power, axis=1)
    yproj = np.sum(power, axis=0)
    
    transverse_grid = np.linspace(-1,1,ncar) * dgrid * 1e6
    ts = dt * np.arange(nslice); ts -= np.mean(ts); temporal_grid = ts * 1e15
    
    if slice == -1:
        xs = temporal_grid; ys = transverse_grid
        xlabel = 'Time (fs)'; ylabel = 'y (um)'; xu = 'fs'; yu = 'um'; xn = 't'; yn = 'y'
    elif slice == -2:
        xs = temporal_grid; ys = transverse_grid
        xlabel = 'Time (fs)'; ylabel = 'x (um)'; xu = 'fs'; yu = 'um'; xn = 't'; yn = 'x'
    else:
        xs = transverse_grid; ys = transverse_grid
        xlabel = 'x (um)'; ylabel = 'y (um)'; xu = 'um'; yu = 'um'; xn = 'x'; yn = 'y'
    xmean = np.dot(xs, xproj) / norm
    ymean = np.dot(ys, yproj) / norm
    xrms = np.sqrt(np.dot(xs**2, xproj) / norm - xmean**2)
    yrms = np.sqrt(np.dot(ys**2, yproj) / norm - ymean**2)
    dx = xs[1] - xs[0]; dy = ys[1] - ys[0]
    xfwhm = fwhm(xproj) * dx
    yfwhm = fwhm(yproj) * dy
    energy_uJ = norm*dt * 1e6
    
    ndecimals = 1
    xmean = np.around(xmean, ndecimals); ymean = np.around(ymean, ndecimals)
    xrms = np.around(xrms, ndecimals); yrms = np.around(yrms, ndecimals)
    xfwhm = np.around(xfwhm, ndecimals)[0]; yfwhm = np.around(yfwhm, ndecimals)[0]
    energy_uJ = np.around(energy_uJ, ndecimals)
    #print('norm =',norm,'   x,y mean =',xmean,ymean, '   x,y rms =', xrms,yrms, '   wx,wy =', 2*xrms,2*yrms)
    if ndecimals == 0:
        xmean = int(xmean); ymean = int(ymean)
        xrms = int(xrms); yrms = int(yrms)
        energy_uJ = int(energy_uJ)
        try:
            xfwhm = int(xfwhm); yfwhm = int(yfwhm)
        except:
            pass
    
    
    #xmean *= 1e6; ymean *= 1e6; xrms *= 1e6; yrms *= 1e6;
    print('norm =',norm,'   ','energy =',energy_uJ,' uJ   ',xn,',',yn,' mean =',xmean,xu,',',ymean, yu,'    ',xn,',',yn,' rms =', xrms,xu,',',yrms, yu, '    w'+xn,', w'+yn,'=', 2*xrms,xu,',',2*yrms,yu,'   ',xn,',',yn,' fwhm =',xfwhm,xu,',',yfwhm, yu)
#     annotation1 = 'energy '+'{:e}'.format(energy_uJ)+' uJ\n'
    annotation1 = 'energy '+str(energy_uJ)+' uJ\n'
    annotation1 += yn+' mean '+str(ymean)+' '+yu+'\n'
    annotation1 += yn+' rms '+str(yrms)+' '+yu+'\n'
    annotation1 += yn+' fwhm '+str(yfwhm)+' '+yu
    annotation2 = xn+' mean '+str(xmean)+' '+xu+'\n'
    annotation2 += xn+' rms '+str(xrms)+' '+xu+'\n'
    annotation2 += xn+' fwhm '+str(xfwhm)+' '+xu
    
    aspect = (min(xs)-max(xs)) / (min(ys)-max(ys))
    # if ax is defined, then collecting plots
    showPlotQ &= (ax == None); savePlotQ &= (ax == None)
    if ax == None: ax = plt.gca()
    if plotPowerQ:
        ax.plot(xs,xproj)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Power (W)')
#         annotation2 = 'energy '+'{:e}'.format(energy_uJ)+' uJ\n' + annotation2
        annotation2 = 'energy '+str(energy_uJ)+' uJ\n' + annotation2
        ax.text(min(xs)+0.01*(max(xs)-min(xs)),min(xproj)+0.86*(max(xproj)-min(xproj)),annotation2,fontsize=10)
    else:
        ax.imshow(power.T, extent=(min(xs),max(xs),min(ys),max(ys)), origin='lower', aspect=aspect, cmap=cmap)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); plt.tight_layout(); 
        ax.text(min(xs)+0.01*(max(xs)-min(xs)),min(ys)+0.86*(max(ys)-min(ys)),annotation1,fontsize=10)
        ax.text(min(xs)+0.01*(max(xs)-min(xs)),min(ys)+0.01*(max(ys)-min(ys)),annotation2,fontsize=10)
        ax.plot(xs, min(ys)+xproj/max(xproj)*0.15*(max(ys)-min(ys)),'k')
        ax.plot(min(xs)+yproj/max(yproj)*0.15*(max(xs)-min(xs)), ys,'k')
    if logScaleQ:
        ax.set_yscale('log')
    if saveFilename != None and savePlotQ:
        plt.savefig(saveFilename, bbox_inches='tight')
    if showPlotQ:
        plt.show()
    plt.close()
    
    
def plot_fld_power(fld, dt, ax=None, saveFilename=None, showPlotQ=True, savePlotQ=True, logScaleQ=False):
    plot_fld_slice(fld, 400e-6, dt, slice=-1, ax=ax, saveFilename=saveFilename, showPlotQ=showPlotQ, savePlotQ=savePlotQ, plotPowerQ=True, logScaleQ=logScaleQ)
    
def plot_fld_marginalize_3(fld, dgrid, dt, title=None):    
    fig, axs = plt.subplots(1,3)
    fig.suptitle(title)
    plot_fld_slice(fld, dgrid=dgrid, dt=dt, slice=-3, ax=axs[0])
    plot_fld_slice(fld, dgrid=dgrid, dt=dt, slice=-2, ax=axs[1])
    plot_fld_slice(fld, dgrid=dgrid, dt=dt, slice=-1, ax=axs[2])
    plt.tight_layout(); 
    #plt.subplot_tool()
    #plt.subplots_adjust(left=0.1, 
                    #bottom=0.1,  
                    #right=0.9,  
                    #top=0.9,  
                    #wspace=0.4,  
                    #hspace=0.4) 
    plt.show()

def plot_fld_marginalize_t(fld, dgrid = 400.e-6, dt=1e-6/3e8, saveFilename=None, showPlotQ=True, savePlotQ=True):
    plot_fld_slice(fld, dgrid=dgrid, dt=dt, slice=-3, saveFilename=saveFilename, showPlotQ=showPlotQ, savePlotQ=savePlotQ)
    
# https://stackoverflow.com/questions/10917495/matplotlib-imshow-in-3d-plot
# https://stackoverflow.com/questions/30464117/plotting-a-imshow-image-in-3d-in-matplotlib
# voxels (slow?) https://terbium.io/2017/12/matplotlib-3d/
# slider https://www.datacamp.com/community/tutorials/matplotlib-3d-volumetric-data
def plot_fld_3d(fld, dgrid = 400.e-6, dt=1e-6/3e8):

     ########## Switch axes due to the different definition between Genesis 2 and Genesis 4
    fld = np.moveaxis(fld, -1, 0)
    ###########################################################################################
    
    # time grid
    nslice = fld.shape[0]
    ts = dt * np.arange(nslice); ts -= np.mean(ts)
    ts *= 1e15
    print(ts)
    
    # transverse grid
    ncar = fld.shape[1]
    dgridum = dgrid * 1e6
    xs = np.linspace(-1,1,ncar) * dgridum; ys = xs
    xv, yv = np.meshgrid(xs, ys) 
    
    # power vs t
    power_vs_t = np.sum(np.sum(fld, axis=1),axis=1)
    power_vs_t /= np.max(power_vs_t)
    
    # make figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    view_z_along_xaxis = True

    # plot slices
    for s in range(nslice):
    
        # transparency gradient colormap 
        # https://kbkb-wx-python.blogspot.com/2015/12/python-transparent-colormap.html
        import matplotlib.colors as mcolors
        ncontourlevels = 21
        colors = [(1,0,0,c*(c>0)) for c in power_vs_t[s] * np.linspace(-1./(ncontourlevels-1),1,ncontourlevels)]
        my_cmap = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=ncontourlevels)
        my_cmap.set_under(color='k', alpha=0)
    
        #ax.imshow(np.abs(fld[s])**2, zs=ts[s], extent=(-dgridum,dgridum,-dgridum,dgridum), 
                   #origin='lower', interpolation='none', cmap=my_cmap, vmin=.001)
        
        if view_z_along_xaxis:
            cset = ax.contourf(np.abs(fld[s])**2, yv, xv, ncontourlevels, zdir='x', offset = ts[s], cmap=my_cmap)
        else:
            cset = ax.contourf(xv, yv, np.abs(fld[s])**2, ncontourlevels, zdir='z', offset = ts[s], cmap=my_cmap)
    
    if view_z_along_xaxis:
        ax.set_xlim([min(ts),max(ts)])
        ax.set_ylim([min(xs),max(xs)])
        ax.set_zlim([min(ys),max(ys)])
        ax.set_zlabel('y (um)')
        ax.set_ylabel('x (um)')
        ax.set_xlabel('t (fs)')
    else:
        ax.set_zlim([min(ts),max(ts)])
        ax.set_xlabel('x (um)')
        ax.set_ylabel('y (um)')
        ax.set_zlabel('t (fs)')

    plt.tight_layout(); plt.show()
