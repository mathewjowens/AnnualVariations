# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 14:58:38 2023

@author: mathe
"""
import pandas as pd
import numpy as np
import os
import astropy.units as u
from astropy.time import Time, TimeDelta
import datetime as datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py
from scipy import interpolate

import helio_time as htime
import ReadICMElist_CaneRichardson as ICMElist

# set the directory of this file as the working directory
cwd = os.path.abspath(os.path.dirname(__file__))

fig_dir = os.path.join(cwd, 'figures')
data_dir = os.path.join(cwd, 'data')
ephem_file = os.path.join(data_dir, 'ephemeris.hdf5')
icme_file = os.path.join(data_dir, 'List of Richardson_Cane ICMEs Since January1996_2022.csv')

#solar min and max definitions
sai_thresh_low = 0.25
sai_thresh_high = 0.67

#plot limits for the solar wind interdependence plots
vmax = 850
vmin = 250
bmin = 0
bmax = 20
nmin = 0
nmax = 20

nbins = 25 # number of bins for histograms

fsize = 14
mpl.rc("axes", labelsize=fsize)
mpl.rc("ytick", labelsize=fsize)
mpl.rc("xtick", labelsize=fsize)
mpl.rc("legend", fontsize=fsize)

# <codecell> data processing

process64secnow = False

if process64secnow:
    data_res_hrs = 1
    
    #load the 64-second data in
    ace64sec = pd.read_hdf(os.path.join(data_dir, 'ace64sec.h5'))
    
    #average to required res (variable labelled 1hr)
    res_str = str(data_res_hrs) + 'H'
    ace1hr = ace64sec.copy()
    ace1hr.set_index('datetime', inplace=True)
    ace1hr = ace1hr.resample(res_str).mean()
    #correct for PANDAS insane time stamp butchering
    ace1hr.index = ace1hr.index + datetime.timedelta(hours = data_res_hrs/2)
    ace1hr = ace1hr.reset_index()
    del ace64sec
    
    #save the 1hr data
    ace1hr.to_hdf(os.path.join(data_dir, 'ace1hr.h5'), key = 'data', mode = 'w')
else:
    ace1hr = pd.read_hdf(os.path.join(data_dir, 'ace1hr.h5'))


#add an AstroPy Time object for fractional year calculation
ace1hr['Time'] = Time(ace1hr['datetime'])



rs = 696340
#add Earth ephemeris
ephem = h5py.File(ephem_file, 'r')
E_time = Time(ephem['EARTH']['HEEQ']['time'], format='jd').mjd
E_r = ephem['EARTH']['HEEQ']['radius'][:]/rs
E_lat = ephem['EARTH']['HEEQ']['latitude'][:]


#interpolate the Earth ephem on the ACE time step
f = interpolate.interp1d( E_time, E_r, fill_value = np.nan, kind = 'nearest')
ace1hr['Earth_r'] = f(ace1hr['mjd'])
ace1hr['pos_r'] = ace1hr['Earth_r'] - ace1hr['pos_gse_x']/rs

f = interpolate.interp1d( E_time, E_lat, fill_value = np.nan, kind = 'nearest')
ace1hr['Earth_lat'] = f(ace1hr['mjd'])


#compute the frac of year
temp = (ace1hr['Time'].to_numpy())
decyr = np.ones((len(ace1hr)))
for n in range(0,len(ace1hr)):
    this_decyr = temp[n].decimalyear
    decyr[n] = this_decyr - np.floor(this_decyr)

ace1hr['frac_of_yr'] = decyr

#compute the angular variation of ACE about the E-S line
#yz = np.sqrt(ace1hr['pos_gse_y']*ace1hr['pos_gse_y'] + 
#             ace1hr['pos_gse_z']*ace1hr['pos_gse_z'])

ace1hr['pos_lat'] = ace1hr['Earth_lat'] + \
    np.arctan2(ace1hr['pos_gse_z']/rs,ace1hr['pos_r']) * 180/np.pi


def LoadSSN(filepath='null'):
    #(dowload from http://www.sidc.be/silso/DATA/SN_m_tot_V2.0.csv)
    if filepath == 'null':
        filepath= os.environ['DBOX'] + 'Data\\SN_m_tot_V2.0.csv'
        
    col_specification =[(0, 4), (5, 7), (8,16),(17,23),(24,29),(30,35)]
    ssn_df=pd.read_fwf(filepath, colspecs=col_specification,header=None)
    dfdt=np.empty_like(ssn_df[0],dtype=datetime.datetime)
    for i in range(0,len(ssn_df)):
        dfdt[i] = datetime.datetime(int(ssn_df[0][i]),int(ssn_df[1][i]),15)
    #replace the index with the datetime objects
    ssn_df['datetime']=dfdt
    ssn_df['ssn']=ssn_df[3]
    ssn_df['mjd'] = htime.datetime2mjd(dfdt)
    #delete the unwanted columns
    ssn_df.drop(0,axis=1,inplace=True)
    ssn_df.drop(1,axis=1,inplace=True)
    ssn_df.drop(2,axis=1,inplace=True)
    ssn_df.drop(3,axis=1,inplace=True)
    ssn_df.drop(4,axis=1,inplace=True)
    ssn_df.drop(5,axis=1,inplace=True)
    
    #add the 13-month running smooth
    window = 13*30
    temp = ssn_df.rolling(str(window)+'D', on='datetime').mean()
    ssn_df['smooth'] = np.interp(ssn_df['mjd'],temp['mjd'],temp['ssn'],
                                              left =np.nan, right =np.nan)
    
    #add in a solar activity index, which normalises the cycle magnitude
    #approx solar cycle length, in months
    nwindow = int(11*12)
    
    #find maximum value in a 1-solar cycle bin centred on current time
    ssn_df['rollingmax'] = ssn_df.rolling(nwindow, center = True).max()['smooth']
    
    #fill the max value at the end of the series
    fillval = ssn_df['rollingmax'].dropna().values[-1]
    ssn_df['rollingmax'] = ssn_df['rollingmax'].fillna(fillval) 
    
    #create a Solar Activity Index, as SSN normalised to the max smoothed value in
    #1-sc window centred on current tim
    ssn_df['sai'] = ssn_df['smooth']/ssn_df['rollingmax']
    
    return ssn_df


ssn = LoadSSN()

#interpolate the SSN on the ACE time step
f = interpolate.interp1d( ssn['mjd'], ssn['smooth'], fill_value = np.nan, kind = 'linear')
ace1hr['ssn'] = f(ace1hr['mjd'])
f = interpolate.interp1d( ssn['mjd'], ssn['sai'], fill_value = np.nan, kind = 'linear')
ace1hr['sai'] = f(ace1hr['mjd'])



#remove ICMEs
ace1hr['Vr nocme'] = ace1hr['Vr']
ace1hr['n_p nocme'] = ace1hr['n_p']
ace1hr['Bmag nocme'] = ace1hr['Bmag']
icmes = ICMElist.ICMElist(icme_file) 
for i in range(0,len(icmes)):
    mask = ((ace1hr['datetime'] >= icmes['Shock_time'][i])
            & (ace1hr['datetime'] < icmes['ICME_end'][i]) )
    ace1hr.loc[mask,'Vr nocme'] = np.nan
    ace1hr.loc[mask,'n_p nocme'] = np.nan
    ace1hr.loc[mask,'Bmag nocme'] = np.nan

# <codecell> SPE functions


#bin up the data
def binxdata(xdata, ydata, bins):
    
    #check whether the number of bins or the bin edges have been specified
    if isinstance(bins,np.ndarray):
        xbinedges=bins
    else:
        xbinedges = np.arange(xdata.min(), xdata.max()+0.01,
                              (xdata.max()-xdata.min())/(bins+1))  
    numbins = len(xbinedges) - 1
        
    xbindata = np.zeros((numbins,4))*np.nan
    for n in range(0,numbins):
        #time at bin centre
        xbindata[n,0] = (xbinedges[n]+xbinedges[n+1])/2
        
        #find the data of interest
        mask =  (xdata >= xbinedges[n]) & (xdata < xbinedges[n+1])
        if np.nansum(mask) > 0:
            xbindata[n,1] = np.nanmean(ydata[mask])
            xbindata[n,2] = np.nanstd(ydata[mask])
            xbindata[n,3] = np.nansum(mask)
            
    return xbindata

def hist2d(xdata, ydata, fighandle=np.nan, axhandle=np.nan,
           nxbins = 10, nybins = 15, xmin = np.nan, xmax = np.nan,
           ymin = np.nan, ymax = np.nan, 
           normcounts = True, plotmedian = True, plotcbar = True, logcounts = True):
    
    if np.isnan(xmin):
        xmin = xdata.min()
    if np.isnan(xmax):
        xmax = xdata.max()
    if np.isnan(ymin):
        ymin = ydata.min()
    if np.isnan(ymax):
        ymax = ydata.max()        
    
    dx = (xmax-xmin)/(nxbins)
    dy = (ymax-ymin)/(nybins)
    
    xbinedges = np.arange(xmin, xmax + dx/1000, dx)  
    ybinedges = np.arange(ymin, ymax + dy/1000, dy) 

    xcentres=(xbinedges[1:] + xbinedges[0:-1]) / 2
    #ycentres=(ybinedges[1:]+ybinedges[0:-1])/2

    xybindata = np.zeros((nybins, nxbins))*np.nan
    ymedian = np.zeros((nxbins))*np.nan
    for x in range(0,nxbins):
     
        #find the data of interest
        mask =  (xdata >= xbinedges[x]) & (xdata < xbinedges[x+1])
        if np.nansum(mask) > 0:
            xybindata[:,x], bin_edges = np.histogram(ydata[mask], bins=ybinedges)
            ymedian[x] = ydata[mask].median()
            if logcounts:
                xybindata[:,x] = np.log10(xybindata[:,x])
            #normalise
            if normcounts:
                N = np.sum(xybindata[:,x])
                xybindata[:,x] = xybindata[:,x] / N

    #x, y = np.meshgrid(xcentres,  ycentres)
    
    # if no fig and axis handles are given, create a new figure
    if isinstance(fighandle, float):
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = fighandle
        ax = axhandle
        
     
    pcol = ax.pcolor(xbinedges,ybinedges,xybindata)
    
    if plotmedian:
        ax.plot(xcentres,ymedian,'w')
        ax.plot(xcentres,ymedian,'k+')
        
    if plotcbar:
        #fig.subplots_adjust(bottom=0.5)
        clabel = 'Counts'
        if logcounts:
            clabel = r'$\log_{10}$ (counts)'
        fig.colorbar(pcol, ax = ax, orientation='vertical', label=clabel)
            
            
    
    return fig, ax, pcol

def contour2d(xdata, ydata, fighandle=np.nan, axhandle=np.nan,
           nxbins = 10, nybins = 15, xmin = np.nan, xmax = np.nan,
           ymin = np.nan, ymax = np.nan, logcounts = True,
           plotcbar = True, plotmedian = True):
    
    if np.isnan(xmin):
        xmin = xdata.min()
    if np.isnan(xmax):
        xmax = xdata.max()
    if np.isnan(ymin):
        ymin = ydata.min()
    if np.isnan(ymax):
        ymax = ydata.max()        
    
    dx = (xmax-xmin)/(nxbins)
    dy = (ymax-ymin)/(nybins)
    
    xbinedges = np.arange(xmin, xmax + dx/1000, dx)  
    ybinedges = np.arange(ymin, ymax + dy/1000, dy) 

    xcentres=(xbinedges[1:] + xbinedges[0:-1]) / 2
    ycentres=(ybinedges[1:]+ybinedges[0:-1])/2

    xybindata = np.zeros((nybins, nxbins))*np.nan
    ymedian = np.zeros((nxbins))*np.nan
    for x in range(0,nxbins):
     
        #find the data of interest
        mask =  (xdata >= xbinedges[x]) & (xdata < xbinedges[x+1])
        if np.nansum(mask) > 0:
            xybindata[:,x], bin_edges = np.histogram(ydata[mask], bins=ybinedges)
            ymedian[x] = ydata[mask].median()
            if logcounts:
                xybindata[:,x] = np.log10(xybindata[:,x])
            

    x, y = np.meshgrid(xcentres,  ycentres)
    
    # if no fig and axis handles are given, create a new figure
    if isinstance(fighandle, float):
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = fighandle
        ax = axhandle
        
     
    pcon = ax.contourf(x,y,xybindata)
    ax.set_ylim([ymin, ymax])
    ax.set_xlim([xmin, xmax])
    
    if plotmedian:
        ax.plot(xcentres,ymedian,'w')
        ax.plot(xcentres,ymedian,'k+')
        
    if plotcbar:
        #fig.subplots_adjust(bottom=0.5)
        clabel = 'Counts'
        if logcounts:
            clabel = r'$\log_{10}$ (counts)'
        fig.colorbar(pcon, ax = ax, orientation='vertical', label=clabel)
            
    
    return fig, ax, pcon


def binned_box_plots(xdata, ydata, fighandle=np.nan, axhandle=np.nan,
           nxbins = 10, xmin = np.nan, xmax = np.nan):
  
    if np.isnan(xmin):
        xmin = xdata.min()
    if np.isnan(xmax):
        xmax = xdata.max()

    dx = (xmax-xmin)/(nxbins)
    
    xbinedges = np.arange(xmin, xmax + dx/1000, dx) 
    xcentres=(xbinedges[1:] + xbinedges[0:-1]) / 2

    # if no fig and axis handles are given, create a new figure
    if isinstance(fighandle, float):
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = fighandle
        ax = axhandle
        
    
    xybindata = []
    for x in range(0,nxbins):
     
        #find the data of interest
        mask =  (xdata >= xbinedges[x]) & (xdata < xbinedges[x+1])
        #if np.nansum(mask) > 0:
        xybindata.append(ydata[mask].dropna())
           
    boxes = ax.boxplot(xybindata, positions = xcentres, widths = dx/2,
               notch=True, patch_artist=True,showfliers=False,whis=1.5)
 
    #ax.tick_params(reset=True)
    #set up the x-axis labels
    ax.set_xlim(xmin - 3*dx/4, xmax + 3*dx/4)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    xtickvals = ax.get_xticks()
    xlabs = []
    for x in xtickvals:
        xlabs.append(str(x))
    ax.set_xticklabels(xlabs)
    


    return fig, ax, boxes

def binned_median_plot(xdata, ydata, fighandle=np.nan, axhandle=np.nan,
           nxbins = 10, xmin = np.nan, xmax = np.nan, fmt=''):
  
    if np.isnan(xmin):
        xmin = xdata.min()
    if np.isnan(xmax):
        xmax = xdata.max()

    dx = (xmax-xmin)/(nxbins)
    
    xbinedges = np.arange(xmin, xmax + dx/1000, dx) 
    xcentres=(xbinedges[1:] + xbinedges[0:-1]) / 2


        
    
    ymedian_upper = np.zeros((nxbins))*np.nan
    ymedian_lower = np.zeros((nxbins))*np.nan
    ymedian = np.zeros((nxbins))*np.nan
    tempfig, tempax = plt.subplots(figsize=(8, 6))
    for x in range(0,nxbins):
     
        #find the data of interest
        mask =  (xdata >= xbinedges[x]) & (xdata < xbinedges[x+1])
        
        #use matplotlib boxplot to get medians and confidence interval
        box = tempax.boxplot(ydata[mask].dropna(), notch=True, showfliers=False, showcaps = False)
        box_y = plt.getp(box['boxes'][0], 'ydata')
        ymedian_upper[x] = box_y[4]
        ymedian_lower[x] = box_y[2]
        ymedian[x] = box_y[3]
        
        #ymedian[x] = np.nanmean(ydata[mask])
        
        plt.close(tempfig)
           
    # if no fig and axis handles are given, create a new figure
    if isinstance(fighandle, float):
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = fighandle
        ax = axhandle
    
    markers, caps, bars = ax.errorbar(xcentres, ymedian, 
                                      yerr = [ymedian - ymedian_lower, ymedian_upper-ymedian],
                                      capsize = 2, fmt=fmt, alpha = 0.5)
    
    # loop through bars and caps and set the alpha value
    [bar.set_alpha(1) for bar in bars]
    [cap.set_alpha(1) for cap in caps]
    #[marker.set_alpha(1) for marker in markers]
    
    ax.set_xlim(xmin - 3*dx/4, xmax + 3*dx/4) 

    return fig, ax

# <codecell> Map everything to 1 AU



@u.quantity_input(v_outer=u.km / u.s)
@u.quantity_input(r_outer=u.solRad)
@u.quantity_input(lon_outer=u.rad)
@u.quantity_input(r_inner=u.solRad)
def map_v_inwards(v_orig, r_orig, lon_orig, r_new):
    """
    Function to map v from r_orig (in rs) to r_inner (in rs) accounting for 
    residual acceleration, but neglecting stream interactions.

    Args:
        v_orig: Solar wind speed at original radial distance. Units of km/s.
        r_orig: Radial distance at original radial distance. Units of km.
        lon_orig: Carrington longitude at original distance. Units of rad
        r_new: Radial distance at new radial distance. Units of km.

    Returns:
        v_new: Solar wind speed mapped from r_orig to r_new. Units of km/s.
        lon_new: Carrington longitude at r_new. Units of rad.
    """

    # Get the acceleration parameters
    alpha = 0.15  # Scale parameter for residual SW acceleration
    rH = (50*u.solRad).to(u.kilometer).value  # Spatial scale parameter for residual SW acceleration
    r_orig = r_orig.to(u.km).value
    r_new = r_new.to(u.km).value
    r_0 = (30*u.solRad).to(u.km).value

    # Compute the 30 rS speed
    v0 = v_orig.value / (1 + alpha * (1 - np.exp(-(r_orig - r_0) / rH)))
    
    #comppute new speed
    vnew = v0 * (1 + alpha * (1 - np.exp(-(r_new - r_0) / rH)))

    # Work out the longitudinal shift
    phi_new = 0

    return vnew * u.km / u.s, phi_new * u.rad


rref = 215

# scale density by r^2
ace1hr['n_p scaled'] = ace1hr['n_p']*ace1hr['pos_r']*ace1hr['pos_r']/rref/rref

#scale speed using the acceleration equation
vorig = ace1hr['Vr'].to_numpy() * u.km/u.s
rorig = ace1hr['pos_r'].to_numpy() * u.solRad
lon = ace1hr['Earth_lat'].to_numpy() *0 *u.deg
rnew = (ace1hr['pos_r'].to_numpy() *0 +rref)*u.solRad
Bmag = ace1hr['Bmag'].to_numpy()

v_215, phi = map_v_inwards(vorig, rorig, lon, rnew)
ace1hr['Vr scaled'] = v_215.value

#scale B by ideal Parker spiral value
sidereal_period = 25.38 * 24*60*60  # Solar sidereal rotation period
#compute the Parker spiral angle
phi_parker_r = np.arctan(2*np.pi * rorig.to(u.km).value
                                    / (sidereal_period*vorig.value))
#compute Br from Bmag and the spiral angle
Br_parker_r = Bmag * np.cos(phi_parker_r)
#scale to reference distance
Br_parker_215 = Br_parker_r* rorig*rorig/rref/rref
#compute the Parker spiral at teh reference distance
phi_parker_215 = np.arctan(2*np.pi * rnew.to(u.km).value
                                    / (sidereal_period*v_215.value))
#computer Bmag at the reference distance
ace1hr['Bmag scaled'] = Br_parker_215 / np.cos(phi_parker_215 )


#compute ram pressure
ace1hr['rho'] = ace1hr['n_p']*100*100*100*1.67262192e-27 # kg/m^3
ace1hr['Pdyn'] = ace1hr['rho'] * ace1hr['Vr'] *1000 * ace1hr['Vr'] *1000
ace1hr['Pdyn'] = ace1hr['Pdyn'] /1e-9 #nPa

ace1hr['rho nocme'] = ace1hr['n_p nocme']*100*100*100*1.67262192e-27 # kg/m^3
ace1hr['Pdyn nocme'] = ace1hr['rho nocme'] * ace1hr['Vr nocme'] *1000 * ace1hr['Vr nocme'] *1000
ace1hr['Pdyn nocme'] = ace1hr['Pdyn nocme'] /1e-9 #nPa


ace1hr['P_B'] = ace1hr['Bmag'] *ace1hr['Bmag']*1e-9 * 1e-9/(2*1.26e-6)
ace1hr['P_B'] = ace1hr['P_B'] /1e-9 #nPa

#include alphas
ace1hr['Pdyn_alpha'] = ace1hr['Pdyn'] + 4* ace1hr['ratio_a2p'] * ace1hr['Pdyn']

#compute scaled ram pressure
ace1hr['rho scaled'] = ace1hr['n_p scaled']*100*100*100*1.67262192e-27 # kg/m^3
ace1hr['Pdyn scaled'] = ace1hr['rho scaled'] * ace1hr['Vr scaled'] *1000 * ace1hr['Vr scaled'] *1000
ace1hr['Pdyn scaled'] = ace1hr['Pdyn scaled'] /1e-9 #nPa

#compute coupling function
a = 0.3
ace1hr['Pinput'] = (  pow(ace1hr['Bmag']*1e-9, 2*a) 
                    * pow(ace1hr['rho'], (2/3  - a)) 
                    * pow(ace1hr['Vr']*1000, 7/3 - 2*a) 
                    )

ace1hr['Pinput scaled'] = (  pow(ace1hr['Bmag scaled']*1e-9, 2*a) 
                    * pow(ace1hr['rho scaled'], (2/3  - a)) 
                    * pow(ace1hr['Vr scaled']*1000, 7/3 - 2*a) 
                    )

ace1hr['Pinput nocme'] = (  pow(ace1hr['Bmag nocme']*1e-9, 2*a) 
                    * pow(ace1hr['rho nocme'], (2/3  - a)) 
                    * pow(ace1hr['Vr nocme']*1000, 7/3 - 2*a) 
                    )
# <codecell> Orbital and SSN time series plots

# plt.figure()
# plt.plot(ace1hr['datetime'],ace1hr['pos_gse_x'],label='GSE X')
# plt.plot(ace1hr['datetime'],ace1hr['pos_gse_y'],label='GSE Y')
# plt.plot(ace1hr['datetime'],ace1hr['pos_gse_z'],label='GSE Z')
# plt.legend()



fig, axs = plt.subplots(nrows = 4, ncols = 1, figsize=(7, 10))

axs[0].plot(ace1hr['datetime'], ace1hr['pos_lat'],'r', label='ACE')
axs[0].plot(ace1hr['datetime'], ace1hr['Earth_lat'] , 'k', label='EARTH')
axs[0].set_ylabel(r'$\theta$, helio latitude [deg]')
axs[0].text(0.03,0.07,'(a)', fontsize = 14, transform=axs[0].transAxes, backgroundcolor = 'w')
#axs[0].set_ylim(bottom=0)
axs[0].legend()
xx = axs[0].get_xlim()
axs[0].plot(xx, [0,0], 'k')
axs[0].set_xlim(xx)

axs[1].plot(ace1hr['datetime'], ace1hr['pos_r'], 'r', label='ACE')
axs[1].plot(ace1hr['datetime'], ace1hr['Earth_r'] , 'k', label='EARTH')
axs[1].set_ylabel(r'$R$, radial distance [$r_\odot$]')
#axs[1].set_ylim(bottom=0)
axs[1].legend()
axs[1].text(0.03,0.07,'(b)', fontsize = 14, transform=axs[1].transAxes, backgroundcolor = 'w')
axs[1].set_xlim(xx)

#plot the fractional variation of r with time
axs[2].plot(ace1hr['datetime'], (ace1hr['Earth_r']-ace1hr['pos_r'])/ace1hr['Earth_r'], 'k')
axs[2].set_ylabel(r'$R _{AE} / R_{ES}$')
axs[2].set_ylim(bottom=0)
axs[2].text(0.03,0.07,'(c)', fontsize = 14, transform=axs[2].transAxes, backgroundcolor = 'w')
axs[2].set_xlim(xx)
#axs[0].set_ylim(bottom=0)

axs[3].plot(ace1hr['datetime'], ace1hr['ssn']/200, 'k', label = 'SSN/200')
axs[3].plot(ace1hr['datetime'], ace1hr['sai'], 'r', label = 'SAI')

axs[3].legend(fontsize = 14, loc = 'upper right')
axs[3].plot([ace1hr['datetime'][0], ace1hr['datetime'][len(ace1hr)-1]], 
            [sai_thresh_low, sai_thresh_low], 'r--')
axs[3].plot([ace1hr['datetime'][0], ace1hr['datetime'][len(ace1hr)-1]], 
            [sai_thresh_high, sai_thresh_high], 'r--')
axs[3].text(0.03,0.07,'(d)', fontsize = 14, transform=axs[3].transAxes, backgroundcolor = 'w')
axs[3].set_ylabel(r'Solar activity')
axs[3].set_xlabel('Year')
axs[3].set_xlim(xx)



#save the figure
fig.set_tight_layout(True)
fig.savefig( os.path.join(fig_dir, 'orbit_timeseries.pdf'))  


# <codecell> Orbital param SPE

N_bins_orbit = 100

datachunk = ace1hr 

xdata = ace1hr['frac_of_yr']
ydata = ace1hr['Earth_r']
rE_binned = binxdata(xdata, ydata, N_bins_orbit)
xdata = ace1hr['frac_of_yr']
ydata = ace1hr['pos_r']
rACE_binned = binxdata(xdata, ydata, N_bins_orbit)

xdata = ace1hr['frac_of_yr']
ydata = ace1hr['Earth_lat']
latE_binned = binxdata(xdata, ydata, N_bins_orbit)
xdata = ace1hr['frac_of_yr']
ydata = ace1hr['pos_lat']
latACE_binned = binxdata(xdata, ydata, N_bins_orbit)

xdata = ace1hr['frac_of_yr']
ydata = np.abs(ace1hr['Earth_lat'])
latE_abs_binned = binxdata(xdata, ydata, N_bins_orbit)
xdata = ace1hr['frac_of_yr']
ydata = np.abs(ace1hr['pos_lat'])
latACE_abs_binned = binxdata(xdata, ydata, N_bins_orbit)

fig, axs = plt.subplots(nrows = 3, ncols = 1, figsize=(6, 8.5))

axs[0].set_xticks([0, 0.25, 0.5, 0.75, 1])
axs[0].xaxis.set_minor_locator(ticker.AutoMinorLocator())
axs[0].errorbar(rE_binned[:,0], rE_binned[:,1], fmt='k', capsize = 2, yerr = rE_binned[:,2],
                label = 'EARTH')
axs[0].errorbar(rACE_binned[:,0], rACE_binned[:,1], fmt='r', capsize = 2, yerr = rACE_binned[:,2],
                label = 'ACE')
axs[0].set_ylabel(r'$R$, radial distance [$r_\odot$]')
axs[0].text(0.03,0.05,'(a)', fontsize = 14, transform=axs[0].transAxes, backgroundcolor = 'w')
axs[0].legend()
xx = axs[0].get_xlim(); 
axs[0].set_xlim(xx)
yy = axs[0].get_ylim(); axs[0].plot([0.5, 0.5] ,yy ,'k')

axs[1].set_xticks([0, 0.25, 0.5, 0.75, 1])
axs[1].xaxis.set_minor_locator(ticker.AutoMinorLocator())
axs[1].errorbar(latE_binned[:,0], latE_binned[:,1], fmt='k',capsize = 2, yerr = latE_binned[:,2],
                label = 'EARTH')
axs[1].errorbar(latACE_binned[:,0], latACE_binned[:,1], fmt='r', capsize = 2, yerr = latACE_binned[:,2],
                label = 'ACE')
axs[1].set_ylabel(r'$\theta$, helio latitude [deg]')
axs[1].text(0.03,0.05,'(b)', fontsize = 14, transform=axs[1].transAxes, backgroundcolor = 'w')
axs[1].legend()
axs[1].plot(xx, [0,0], 'k')
axs[1].set_xlim(xx)
yy = axs[1].get_ylim(); axs[1].plot([0.5, 0.5] ,yy ,'k')


axs[2].set_xticks([0, 0.25, 0.5, 0.75, 1])
axs[2].xaxis.set_minor_locator(ticker.AutoMinorLocator())
axs[2].errorbar(latE_abs_binned[:,0], latE_abs_binned[:,1], fmt='k', capsize = 2, yerr = latE_abs_binned[:,2],
                label = 'EARTH')
axs[2].errorbar(latACE_abs_binned[:,0], latACE_abs_binned[:,1], fmt='r', capsize = 2, yerr = latACE_abs_binned[:,2],
                label = 'ACE')
axs[2].set_ylabel(r'$|\theta|$, absolute' '\n' 'helio latitude [deg]')
axs[2].text(0.03,0.05,'(c)', fontsize = 14, transform=axs[2].transAxes, backgroundcolor = 'w')
axs[2].legend(loc = 'upper right')
axs[2].set_xlabel('Fraction of year, $F$')
axs[2].set_xlim(xx)
yy = axs[2].get_ylim(); axs[2].plot([0.5, 0.5] ,yy ,'k')

#save the figure
fig.set_tight_layout(True)
fig.savefig( os.path.join(fig_dir,'orbit_SPE.pdf')) 
# <codecell> r and lat interdependence

#cut out the datachunk of interest
mask = (ace1hr['ssn'] <= 30000)
datachunk = ace1hr.loc[mask] 

fig, axs = plt.subplots(nrows = 2, ncols = 1, figsize=(6, 8))

dx = 0.5
xdata = np.abs(datachunk['Earth_r'])
ydata = (datachunk['Earth_lat'])
hist2d(xdata, ydata, nxbins = 25, nybins = 25, fighandle = fig, axhandle = axs[0],
                 xmin = xdata.min() - dx, xmax = xdata.max() + dx,
                 plotmedian = False, logcounts = True, normcounts = False, plotcbar = True)
axs[0].set_ylabel(r'$\theta$, helio latitude [deg]')
axs[0].text(0.02,0.93,'(a)', fontsize = 14, transform=axs[0].transAxes, backgroundcolor = 'w')

xdata = np.abs(datachunk['Earth_r'])
ydata = np.abs(datachunk['Earth_lat'])
fig, ax, pcol = hist2d(xdata, ydata, nxbins = 25, nybins = 25, fighandle = fig, axhandle = axs[1],
                 xmin = xdata.min() - dx, xmax = xdata.max() + dx,
                 ymin = 0, plotmedian = False, logcounts = True, normcounts = False, plotcbar = True)
axs[1].set_ylabel(r'$|\theta|$, absolute helio latitude [deg]')
axs[1].set_xlabel(r'$R$, radial distance [$r_\odot$]')
axs[1].text(0.02,0.93,'(b)', fontsize = 14, transform=axs[1].transAxes, backgroundcolor = 'w')

#cbar = fig.colorbar(pcol, ax=axs, shrink=0.6)
#cbar.set_label('log(counts)')

fig.set_tight_layout(True)
fig.savefig( os.path.join(fig_dir, 'orbit_r_lat_interdependence.pdf')) 

# <codecell> Solar wind param interdependence



mask = (ace1hr['sai'] >= 0) 
datachunk = ace1hr.loc[mask] 

# fig, axs = plt.subplots(nrows = 3, ncols = 1, figsize=(6, 10))

# xdata = np.abs(datachunk['Vr'])
# ydata = np.abs(datachunk['n_p'])
# contour2d(xdata, ydata, nxbins = nbins, nybins = nbins, fighandle = fig, axhandle = axs[0],
#                   xmin = vmin, xmax = vmax, ymin = nmin, ymax = nmax)
# axs[0].set_xlabel(r'$V_{SW}$ [km s$^{-1}$]')
# axs[0].set_ylabel(r'$n_{P}$ [cm$^{-3}$]')
# axs[0].text(0.02,0.9,'(a)', fontsize = 14, transform=axs[0].transAxes, backgroundcolor = 'w')

# xdata = np.abs(datachunk['Vr'])
# ydata = np.abs(datachunk['Bmag'])
# contour2d(xdata, ydata, nxbins = nbins, nybins = nbins, fighandle = fig, axhandle = axs[1],
#                   xmin = vmin, xmax = vmax, ymin = bmin, ymax = bmax)
# axs[1].set_xlabel(r'$V_{SW}$ [km s$^{-1}$]')
# axs[1].set_ylabel(r'$|$B$|$ [nT]')
# axs[1].text(0.02,0.9,'(b)', fontsize = 14, transform=axs[1].transAxes, backgroundcolor = 'w')

# xdata = np.abs(datachunk['n_p'])
# ydata = np.abs(datachunk['Bmag'])
# contour2d(xdata, ydata, nxbins = nbins, nybins = nbins, fighandle = fig, axhandle = axs[2],
#                   xmin = nmin, xmax = nmax, ymin = bmin, ymax = bmax)
# axs[2].set_xlabel(r'$n_{P}$ [cm$^{-3}$]')
# axs[2].set_ylabel(r'$|$B$|$ [nT]')
# axs[2].text(0.02,0.9,'(c)', fontsize = 14, transform=axs[2].transAxes, backgroundcolor = 'w')
# fig.set_tight_layout(True)
# fig.savefig(fig_dir + 'solarwindparam_interdependence.pdf') 



# fig, axs = plt.subplots(nrows = 3, ncols = 1, figsize=(6, 10))

# xdata = np.abs(datachunk['Vr'])
# ydata = np.abs(datachunk['n_p'])
# binned_box_plots(xdata, ydata, nxbins = nbins,  fighandle = fig, axhandle = axs[0],
#                  xmin = vmin, xmax = vmax, )
# axs[0].set_xlabel(r'$V_{SW}$ [km s$^{-1}$]')
# axs[0].set_ylabel(r'$n_{P}$ [cm$^{-3}$]')
# axs[0].text(0.02,0.9,'(a)', fontsize = 14, transform=axs[0].transAxes, backgroundcolor = 'w')

# xdata = np.abs(datachunk['Vr'])
# ydata = np.abs(datachunk['Bmag'])
# binned_box_plots(xdata, ydata, nxbins = nbins,  fighandle = fig, axhandle = axs[1],
#                  xmin = vmin, xmax = vmax, )
# axs[1].set_xlabel(r'$V_{SW}$ [km s$^{-1}$]')
# axs[1].set_ylabel(r'$|$B$|$ [nT]')
# axs[1].text(0.02,0.9,'(b)', fontsize = 14, transform=axs[1].transAxes, backgroundcolor = 'w')

# xdata = np.abs(datachunk['n_p'])
# ydata = np.abs(datachunk['Bmag'])
# binned_box_plots(xdata, ydata, nxbins = nbins, fighandle = fig, axhandle = axs[2],
#                  xmin = nmin, xmax = nmax, )
# axs[2].set_xlabel(r'$n_{P}$ [cm$^{-3}$]')
# axs[2].set_ylabel(r'$|$B$|$ [nT]')
# axs[2].text(0.02,0.9,'(c)', fontsize = 14, transform=axs[2].transAxes, backgroundcolor = 'w')

# fig.set_tight_layout(True)
# fig.savefig(fig_dir + 'solarwindparam_interdependence_boxplots.pdf') 


fig, axs = plt.subplots(nrows = 3, ncols = 1, figsize=(6, 10))

xdata = np.abs(datachunk['Vr'])
ydata = np.abs(datachunk['n_p'])
binned_median_plot(xdata, ydata, nxbins = nbins,  fighandle = fig, axhandle = axs[0],
                 xmin = vmin, xmax = vmax, fmt='k' )
xdata = np.abs(datachunk['Vr scaled'])
ydata = np.abs(datachunk['n_p scaled'])
binned_median_plot(xdata, ydata, nxbins = nbins,  fighandle = fig, axhandle = axs[0],
                 xmin = vmin, xmax = vmax, fmt='r' )
xdata = np.abs(datachunk['Vr nocme'])
ydata = np.abs(datachunk['n_p nocme'])
binned_median_plot(xdata, ydata, nxbins = nbins,  fighandle = fig, axhandle = axs[0],
                 xmin = vmin, xmax = vmax, fmt='b' )
axs[0].set_xlabel(r'$V_{R}$ [km s$^{-1}$]')
axs[0].set_ylabel(r'$n_{P}$ [cm$^{-3}$]')
axs[0].text(0.02,0.9,'(a)', fontsize = 14, transform=axs[0].transAxes, backgroundcolor = 'w')
axs[0].legend([r'Observed', r'$R$-scaled', 'No ICMEs'], loc = 'upper right')

xdata = np.abs(datachunk['Vr'])
ydata = np.abs(datachunk['Bmag'])
binned_median_plot(xdata, ydata, nxbins = nbins,  fighandle = fig, axhandle = axs[1],
                 xmin = vmin, xmax = vmax , fmt='k')
xdata = np.abs(datachunk['Vr scaled'])
ydata = np.abs(datachunk['Bmag scaled'])
binned_median_plot(xdata, ydata, nxbins = nbins,  fighandle = fig, axhandle = axs[1],
                 xmin = vmin, xmax = vmax , fmt='r')
xdata = np.abs(datachunk['Vr nocme'])
ydata = np.abs(datachunk['Bmag nocme'])
binned_median_plot(xdata, ydata, nxbins = nbins,  fighandle = fig, axhandle = axs[1],
                 xmin = vmin, xmax = vmax , fmt='b')
axs[1].set_xlabel(r'$V_{R}$ [km s$^{-1}$]')
axs[1].set_ylabel(r'$|$B$|$ [nT]')
axs[1].text(0.02,0.9,'(b)', fontsize = 14, transform=axs[1].transAxes, backgroundcolor = 'w')

xdata = np.abs(datachunk['n_p'])
ydata = np.abs(datachunk['Bmag'])
binned_median_plot(xdata, ydata, nxbins = nbins, fighandle = fig, axhandle = axs[2],
                 xmin = nmin, xmax = nmax , fmt='k')
xdata = np.abs(datachunk['n_p scaled'])
ydata = np.abs(datachunk['Bmag scaled'])
binned_median_plot(xdata, ydata, nxbins = nbins, fighandle = fig, axhandle = axs[2],
                 xmin = nmin, xmax = nmax , fmt='r')
xdata = np.abs(datachunk['n_p nocme'])
ydata = np.abs(datachunk['Bmag nocme'])
binned_median_plot(xdata, ydata, nxbins = nbins, fighandle = fig, axhandle = axs[2],
                 xmin = nmin, xmax = nmax , fmt='b')
axs[2].set_xlabel(r'$n_{P}$ [cm$^{-3}$]')
axs[2].set_ylabel(r'$|$B$|$ [nT]')
axs[2].text(0.02,0.9,'(c)', fontsize = 14, transform=axs[2].transAxes, backgroundcolor = 'w')

fig.set_tight_layout(True)
fig.savefig( os.path.join(fig_dir, 'solarwindparam_interdependence_medians.pdf')) 













# #flip x and y params

# fig, axs = plt.subplots(nrows = 3, ncols = 1, figsize=(6, 10))

# ydata = np.abs(datachunk['Vr'])
# xdata = np.abs(datachunk['n_p'])
# binned_median_plot(xdata, ydata, nxbins = nbins,  fighandle = fig, axhandle = axs[0],
#                  xmin = nmin, xmax = nmax, fmt='k' )
# ydata = np.abs(datachunk['Vr scaled'])
# xdata = np.abs(datachunk['n_p scaled'])
# binned_median_plot(xdata, ydata, nxbins = nbins,  fighandle = fig, axhandle = axs[0],
#                  xmin = nmin, xmax = nmax, fmt='r' )
# ydata = np.abs(datachunk['Vr nocme'])
# xdata = np.abs(datachunk['n_p nocme'])
# binned_median_plot(xdata, ydata, nxbins = nbins,  fighandle = fig, axhandle = axs[0],
#                  xmin = nmin, xmax = nmax, fmt='b' )
# axs[0].set_ylabel(r'$V_{R}$ [km s$^{-1}$]')
# axs[0].set_xlabel(r'$n_{P}$ [cm$^{-3}$]')
# axs[0].text(0.02,0.9,'(a)', fontsize = 14, transform=axs[0].transAxes, backgroundcolor = 'w')
# axs[0].legend([r'Observed', r'$R$-scaled', 'No ICMEs'], loc = 'upper right')

# ydata = np.abs(datachunk['Vr'])
# xdata = np.abs(datachunk['Bmag'])
# binned_median_plot(xdata, ydata, nxbins = nbins,  fighandle = fig, axhandle = axs[1],
#                  xmin = bmin, xmax = bmax , fmt='k')
# ydata = np.abs(datachunk['Vr scaled'])
# xdata = np.abs(datachunk['Bmag scaled'])
# binned_median_plot(xdata, ydata, nxbins = nbins,  fighandle = fig, axhandle = axs[1],
#                  xmin = bmin, xmax = bmax , fmt='r')
# ydata = np.abs(datachunk['Vr nocme'])
# xdata = np.abs(datachunk['Bmag nocme'])
# binned_median_plot(xdata, ydata, nxbins = nbins,  fighandle = fig, axhandle = axs[1],
#                  xmin = bmin, xmax = bmax , fmt='b')
# axs[1].set_ylabel(r'$V_{R}$ [km s$^{-1}$]')
# axs[1].set_xlabel(r'$|$B$|$ [nT]')
# axs[1].text(0.02,0.9,'(b)', fontsize = 14, transform=axs[1].transAxes, backgroundcolor = 'w')

# ydata = np.abs(datachunk['n_p'])
# xdata = np.abs(datachunk['Bmag'])
# binned_median_plot(xdata, ydata, nxbins = nbins, fighandle = fig, axhandle = axs[2],
#                  xmin = bmin, xmax = bmax , fmt='k')
# ydata = np.abs(datachunk['n_p scaled'])
# xdata = np.abs(datachunk['Bmag scaled'])
# binned_median_plot(xdata, ydata, nxbins = nbins, fighandle = fig, axhandle = axs[2],
#                  xmin = bmin, xmax = bmax , fmt='r')
# ydata = np.abs(datachunk['n_p nocme'])
# xdata = np.abs(datachunk['Bmag nocme'])
# binned_median_plot(xdata, ydata, nxbins = nbins, fighandle = fig, axhandle = axs[2],
#                  xmin = bmin, xmax = bmax , fmt='b')
# axs[2].set_ylabel(r'$n_{P}$ [cm$^{-3}$]')
# axs[2].set_xlabel(r'$|$B$|$ [nT]')
# axs[2].text(0.02,0.9,'(c)', fontsize = 14, transform=axs[2].transAxes, backgroundcolor = 'w')

# fig.set_tight_layout(True)
# fig.savefig(fig_dir + 'solarwindparam_interdependence_medians_inv.pdf') 

# <codecell> solar wind variations with r and lat



def plotparam(datachunk, param, ymin, ymax, ylab, nbins, ssn_thresh):
    fig, axs = plt.subplots(nrows = 3, ncols = 3, figsize=(10, 10))
    
    mask = (ace1hr['ssn'] <= 30000)
    datachunk = ace1hr.loc[mask] 
    
    xdata = np.abs(datachunk['pos_r'])
    ydata = np.abs(datachunk[param])
    contour2d(xdata, ydata, nxbins = nbins, nybins = nbins, fighandle = fig, axhandle = axs[0,0],
                      ymin = ymin, ymax = ymax,  plotcbar = False)
    axs[0,0].set_title('All data')
    axs[0,0].set_ylabel(ylab)
    axs[0,0].set_xlabel(r'R [$r_\odot$]')
    axs[0,0].text(0.03,0.9,'(a)', fontsize = 14, transform=axs[0,0].transAxes, backgroundcolor = 'w')
    
    xdata = (datachunk['pos_lat'])
    ydata = np.abs(datachunk[param])
    contour2d(xdata, ydata, nxbins = nbins, nybins = nbins, fighandle = fig, axhandle = axs[1,0],
                     ymin = ymin, ymax = ymax,  plotcbar = False)
    axs[1,0].set_ylabel(ylab)
    axs[1,0].set_xlabel(r'$\theta$ [deg]')
    axs[1,0].text(0.03,0.9,'(d)', fontsize = 14, transform=axs[1,0].transAxes, backgroundcolor = 'w')
    
    
    
    xdata = np.abs(datachunk['pos_lat'])
    ydata = np.abs(datachunk[param])
    contour2d(xdata, ydata, nxbins = nbins, nybins = nbins, fighandle = fig, axhandle = axs[2,0],
                     xmin = 0, ymin = ymin, ymax = ymax,  plotcbar = False)
    axs[2,0].set_ylabel(ylab)
    axs[2,0].set_xlabel(r'$|\theta|$ [deg]')
    axs[2,0].text(0.03,0.9,'(g)', fontsize = 14, transform=axs[2,0].transAxes, backgroundcolor = 'w')
    
    
    
    
    mask = (ace1hr['ssn'] <= ssn_thresh)
    datachunk = ace1hr.loc[mask] 
    
    xdata = np.abs(datachunk['pos_r'])
    ydata = np.abs(datachunk[param])
    contour2d(xdata, ydata, nxbins = nbins, nybins = nbins, fighandle = fig, axhandle = axs[0,1],
                      ymin = ymin, ymax = ymax,  plotcbar = False)
    axs[0,1].set_title('SSN <= ' +str(ssn_thresh))
    axs[0,1].set_xlabel(r'R [$r_\odot$]')
    axs[0,1].text(0.03,0.9,'(b)', fontsize = 14, transform=axs[0,1].transAxes, backgroundcolor = 'w')
    
    
    xdata = (datachunk['pos_lat'])
    ydata = np.abs(datachunk[param])
    contour2d(xdata, ydata, nxbins = nbins, nybins = nbins, fighandle = fig, axhandle = axs[1,1],
                     ymin = ymin, ymax = ymax,  plotcbar = False)
    axs[1,1].set_xlabel(r'$\theta$ [deg]')
    axs[1,1].text(0.03,0.9,'(d)', fontsize = 14, transform=axs[1,1].transAxes, backgroundcolor = 'w')
    
    
    
    xdata = np.abs(datachunk['pos_lat'])
    ydata = np.abs(datachunk[param])
    contour2d(xdata, ydata, nxbins = nbins, nybins = nbins, fighandle = fig, axhandle = axs[2,1],
                     xmin = 0, ymin = ymin, ymax = ymax,  plotcbar = False)
    axs[2,1].set_xlabel(r'$|\theta|$ [deg]')
    axs[2,1].text(0.03,0.9,'(h)', fontsize = 14, transform=axs[2,1].transAxes, backgroundcolor = 'w')
    
    
    
    mask = (ace1hr['ssn'] > ssn_thresh)
    datachunk = ace1hr.loc[mask] 
    
    xdata = np.abs(datachunk['pos_r'])
    ydata = np.abs(datachunk[param])
    contour2d(xdata, ydata, nxbins = nbins, nybins = nbins, fighandle = fig, axhandle = axs[0,2],
                      ymin = ymin, ymax = ymax,  plotcbar = False)
    axs[0,2].set_title('SSN > ' +str(ssn_thresh))
    axs[0,2].set_xlabel(r'R [$r_\odot$]')
    axs[0,2].text(0.03,0.9,'(c)', fontsize = 14, transform=axs[0,2].transAxes, backgroundcolor = 'w')
    
    xdata = (datachunk['pos_lat'])
    ydata = np.abs(datachunk[param])
    contour2d(xdata, ydata, nxbins = nbins, nybins = nbins, fighandle = fig, axhandle = axs[1,2],
                     ymin = ymin, ymax = ymax,  plotcbar = False)
    axs[1,2].set_xlabel(r'$\theta$ [deg]')
    axs[1,2].text(0.03,0.9,'(f)', fontsize = 14, transform=axs[1,2].transAxes, backgroundcolor = 'w')
    
    
    xdata = np.abs(datachunk['pos_lat'])
    ydata = np.abs(datachunk[param])
    fig, ax,cmap = contour2d(xdata, ydata, nxbins = nbins, nybins = nbins, fighandle = fig, axhandle = axs[2,2],
                     xmin = 0, ymin = ymin, ymax = ymax,  plotcbar = False)
    axs[2,2].set_xlabel(r'$|\theta|$ [deg]')
    axs[2,2].text(0.03,0.9,'(i)', fontsize = 14, transform=axs[2,2].transAxes, backgroundcolor = 'w')
    
    fig.tight_layout(rect=[0, 0.1, 1, 0.95])
    # fig.subplots_adjust(bottom=0.5)
    cax = plt.axes([0.12, 0.06, 0.8, 0.02]) #Left,bottom, length, width
    clb=plt.colorbar(cmap, cax=cax,orientation="horizontal")
    # clb.ax.tick_params(labelsize=8) 
    clb.set_label('log(counts)',fontsize=14)
    
    fig.savefig( os.path.join(fig_dir, param + '_orbitalparams.pdf')) 
    
    
def plotparam_box(datachunk, param, ylab, nbins, ssn_thresh):
    fig, axs = plt.subplots(nrows = 3, ncols = 3, figsize=(10, 10))
    
    mask = (ace1hr['ssn'] <= 30000)
    datachunk = ace1hr.loc[mask] 
    
    xdata = np.abs(datachunk['pos_r'])
    ydata = np.abs(datachunk[param])
    binned_box_plots(xdata, ydata, nxbins = nbins,fighandle = fig, axhandle = axs[0,0])
    axs[0,0].set_title('All data')
    axs[0,0].set_ylabel(ylab)
    axs[0,0].set_xlabel(r'R [$r_\odot$]')
    axs[0,0].text(0.03,0.9,'(a)', fontsize = 14, transform=axs[0,0].transAxes, backgroundcolor = 'w')
    
    xdata = (datachunk['pos_lat'])
    ydata = np.abs(datachunk[param])
    binned_box_plots(xdata, ydata, nxbins = nbins, fighandle = fig, axhandle = axs[1,0])
    axs[1,0].set_ylabel(ylab)
    axs[1,0].set_xlabel(r'$\theta$ [deg]')
    axs[1,0].text(0.03,0.9,'(d)', fontsize = 14, transform=axs[1,0].transAxes, backgroundcolor = 'w')
    
    
    
    xdata = np.abs(datachunk['pos_lat'])
    ydata = np.abs(datachunk[param])
    binned_box_plots(xdata, ydata, nxbins = nbins, fighandle = fig, axhandle = axs[2,0])
    axs[2,0].set_ylabel(ylab)
    axs[2,0].set_xlabel(r'$|\theta|$ [deg]')
    axs[2,0].text(0.03,0.9,'(g)', fontsize = 14, transform=axs[2,0].transAxes, backgroundcolor = 'w')
    
    
    
    
    mask = (ace1hr['ssn'] <= ssn_thresh)
    datachunk = ace1hr.loc[mask] 
    
    xdata = np.abs(datachunk['pos_r'])
    ydata = np.abs(datachunk[param])
    binned_box_plots(xdata, ydata, nxbins = nbins, fighandle = fig, axhandle = axs[0,1])
    axs[0,1].set_title('SSN <= ' +str(ssn_thresh))
    axs[0,1].set_xlabel(r'R [$r_\odot$]')
    axs[0,1].text(0.03,0.9,'(b)', fontsize = 14, transform=axs[0,1].transAxes, backgroundcolor = 'w')
    
    
    xdata = (datachunk['pos_lat'])
    ydata = np.abs(datachunk[param])
    binned_box_plots(xdata, ydata, nxbins = nbins, fighandle = fig, axhandle = axs[1,1])
    axs[1,1].set_xlabel(r'$\theta$ [deg]')
    axs[1,1].text(0.03,0.9,'(d)', fontsize = 14, transform=axs[1,1].transAxes, backgroundcolor = 'w')
    
    
    
    xdata = np.abs(datachunk['pos_lat'])
    ydata = np.abs(datachunk[param])
    binned_box_plots(xdata, ydata, nxbins = nbins, fighandle = fig, axhandle = axs[2,1])
    axs[2,1].set_xlabel(r'$|\theta|$ [deg]')
    axs[2,1].text(0.03,0.9,'(h)', fontsize = 14, transform=axs[2,1].transAxes, backgroundcolor = 'w')
    
    
    
    mask = (ace1hr['ssn'] > ssn_thresh)
    datachunk = ace1hr.loc[mask] 
    
    xdata = np.abs(datachunk['pos_r'])
    ydata = np.abs(datachunk[param])
    binned_box_plots(xdata, ydata, nxbins = nbins, fighandle = fig, axhandle = axs[0,2])
    axs[0,2].set_title('SSN > ' +str(ssn_thresh))
    axs[0,2].set_xlabel(r'R [$r_\odot$]')
    axs[0,2].text(0.03,0.9,'(c)', fontsize = 14, transform=axs[0,2].transAxes, backgroundcolor = 'w')
    
    xdata = (datachunk['pos_lat'])
    ydata = np.abs(datachunk[param])
    binned_box_plots(xdata, ydata, nxbins = nbins, fighandle = fig, axhandle = axs[1,2])
    axs[1,2].set_xlabel(r'$\theta$ [deg]')
    axs[1,2].text(0.03,0.9,'(f)', fontsize = 14, transform=axs[1,2].transAxes, backgroundcolor = 'w')
    
    
    xdata = np.abs(datachunk['pos_lat'])
    ydata = np.abs(datachunk[param])
    binned_box_plots(xdata, ydata, nxbins = nbins, fighandle = fig, axhandle = axs[2,2])
    axs[2,2].set_xlabel(r'$|\theta|$ [deg]')
    axs[2,2].text(0.03,0.9,'(i)', fontsize = 14, transform=axs[2,2].transAxes, backgroundcolor = 'w')
    
    fig.tight_layout(rect=[0, 0.1, 1, 0.95])
    # fig.subplots_adjust(bottom=0.5)
    #cax = plt.axes([0.12, 0.06, 0.8, 0.02]) #Left,bottom, length, width
    #clb=plt.colorbar(cmap, cax=cax,orientation="horizontal")
    # clb.ax.tick_params(labelsize=8) 
   # clb.set_label('log(counts)',fontsize=14)
    
    fig.savefig( os.path.join(fig_dir,param + '_orbitalparams_boxplot.pdf')) 
    
def plotparam_medians(datachunk, param, ylab, nbins, ssn_thresh, fmt = ''):
    fig, axs = plt.subplots(nrows = 3, ncols = 3, figsize=(10, 10))
    
    mask = (ace1hr['ssn'] <= 30000)
    datachunk = ace1hr.loc[mask] 
    
    xdata = np.abs(datachunk['pos_r'])
    ydata = np.abs(datachunk[param])
    binned_median_plot(xdata, ydata, nxbins = nbins,fighandle = fig, axhandle = axs[0,0], fmt=fmt)
    axs[0,0].set_title('All data')
    axs[0,0].set_ylabel(ylab)
    axs[0,0].set_xlabel(r'R [$r_\odot$]')
    axs[0,0].text(0.03,0.9,'(a)', fontsize = 14, transform=axs[0,0].transAxes, backgroundcolor = 'w')
    
    xdata = (datachunk['pos_lat'])
    ydata = np.abs(datachunk[param])
    binned_median_plot(xdata, ydata, nxbins = nbins, fighandle = fig, axhandle = axs[1,0])
    axs[1,0].set_ylabel(ylab)
    axs[1,0].set_xlabel(r'$\theta$ [deg]')
    axs[1,0].text(0.03,0.9,'(d)', fontsize = 14, transform=axs[1,0].transAxes, backgroundcolor = 'w')
    
    
    
    xdata = np.abs(datachunk['pos_lat'])
    ydata = np.abs(datachunk[param])
    binned_median_plot(xdata, ydata, nxbins = nbins, fighandle = fig, axhandle = axs[2,0], fmt=fmt)
    axs[2,0].set_ylabel(ylab)
    axs[2,0].set_xlabel(r'$|\theta|$ [deg]')
    axs[2,0].text(0.03,0.9,'(g)', fontsize = 14, transform=axs[2,0].transAxes, backgroundcolor = 'w')
    
    
    
    
    mask = (ace1hr['ssn'] <= ssn_thresh)
    datachunk = ace1hr.loc[mask] 
    
    xdata = np.abs(datachunk['pos_r'])
    ydata = np.abs(datachunk[param])
    binned_median_plot(xdata, ydata, nxbins = nbins, fighandle = fig, axhandle = axs[0,1], fmt=fmt)
    axs[0,1].set_title('SSN <= ' +str(ssn_thresh))
    axs[0,1].set_xlabel(r'R [$r_\odot$]')
    axs[0,1].text(0.03,0.9,'(b)', fontsize = 14, transform=axs[0,1].transAxes, backgroundcolor = 'w')
    
    
    xdata = (datachunk['pos_lat'])
    ydata = np.abs(datachunk[param])
    binned_median_plot(xdata, ydata, nxbins = nbins, fighandle = fig, axhandle = axs[1,1], fmt=fmt)
    axs[1,1].set_xlabel(r'$\theta$ [deg]')
    axs[1,1].text(0.03,0.9,'(d)', fontsize = 14, transform=axs[1,1].transAxes, backgroundcolor = 'w')
    
    
    
    xdata = np.abs(datachunk['pos_lat'])
    ydata = np.abs(datachunk[param])
    binned_median_plot(xdata, ydata, nxbins = nbins, fighandle = fig, axhandle = axs[2,1], fmt=fmt)
    axs[2,1].set_xlabel(r'$|\theta|$ [deg]')
    axs[2,1].text(0.03,0.9,'(h)', fontsize = 14, transform=axs[2,1].transAxes, backgroundcolor = 'w')
    
    
    
    mask = (ace1hr['ssn'] > ssn_thresh)
    datachunk = ace1hr.loc[mask] 
    
    xdata = np.abs(datachunk['pos_r'])
    ydata = np.abs(datachunk[param])
    binned_median_plot(xdata, ydata, nxbins = nbins, fighandle = fig, axhandle = axs[0,2], fmt=fmt)
    axs[0,2].set_title('SSN > ' +str(ssn_thresh))
    axs[0,2].set_xlabel(r'R [$r_\odot$]')
    axs[0,2].text(0.03,0.9,'(c)', fontsize = 14, transform=axs[0,2].transAxes, backgroundcolor = 'w')
    
    xdata = (datachunk['pos_lat'])
    ydata = np.abs(datachunk[param])
    binned_median_plot(xdata, ydata, nxbins = nbins, fighandle = fig, axhandle = axs[1,2], fmt=fmt)
    axs[1,2].set_xlabel(r'$\theta$ [deg]')
    axs[1,2].text(0.03,0.9,'(f)', fontsize = 14, transform=axs[1,2].transAxes, backgroundcolor = 'w')
    
    
    xdata = np.abs(datachunk['pos_lat'])
    ydata = np.abs(datachunk[param])
    binned_median_plot(xdata, ydata, nxbins = nbins, fighandle = fig, axhandle = axs[2,2], fmt=fmt)
    axs[2,2].set_xlabel(r'$|\theta|$ [deg]')
    axs[2,2].text(0.03,0.9,'(i)', fontsize = 14, transform=axs[2,2].transAxes, backgroundcolor = 'w')
    
    fig.tight_layout(rect=[0, 0.1, 1, 0.95])
    # fig.subplots_adjust(bottom=0.5)
    #cax = plt.axes([0.12, 0.06, 0.8, 0.02]) #Left,bottom, length, width
    #clb=plt.colorbar(cmap, cax=cax,orientation="horizontal")
    # clb.ax.tick_params(labelsize=8) 
   # clb.set_label('log(counts)',fontsize=14)
    
    fig.savefig( os.path.join(fig_dir, param + '_orbitalparams_medians.pdf')) 


## <codecell> solar wind with orbital param plots
# nbins = 10

# param = 'n_p'
# ymin = nmin
# ymax = 3*nmax/4
# ylab = r'$n_{P}$ [cm$^{-3}$]'
# plotparam(datachunk, param, ymin, ymax, ylab, nbins, ssn_thresh)
# plotparam_box(datachunk, param, ylab, nbins, ssn_thresh)
# plotparam_medians(datachunk, param, ylab, nbins, ssn_thresh)

# param = 'Vr'
# ymin = vmin
# ymax = 3*vmax/4
# ylab = r'$V_r$ [km s$^{-1}$]'
# plotparam(datachunk, param, ymin, ymax, ylab, nbins, ssn_thresh)
# plotparam_box(datachunk, param, ylab, nbins, ssn_thresh)
# plotparam_medians(datachunk, param, ylab, nbins, ssn_thresh)

# param = 'Bmag'
# ymin = bmin
# ymax = 3*bmax/4
# ylab = r'$|$B$|$ [nT]'

# plotparam(datachunk, param, ymin, ymax, ylab, nbins, ssn_thresh) 
# plotparam_box(datachunk, param, ylab, nbins, ssn_thresh)
# plotparam_medians(datachunk, param, ylab, nbins, ssn_thresh)

# <codecell> solar wind params condensed

fig, axs = plt.subplots(nrows = 3, ncols = 3, figsize=(10, 10))

#mask = (ace1hr['ssn'] <= 30000)
mask = (ace1hr['sai'] <= 30000)
datachunk = ace1hr.loc[mask] 

param = 'n_p'
ymin = 3.6
ymax = 5.2
ylab = r'$n_{P}$ [cm$^{-3}$]'

xdata = np.abs(datachunk['pos_r'])
ydata = np.abs(datachunk[param])
binned_median_plot(xdata, ydata, nxbins = nbins,fighandle = fig, axhandle = axs[0,0], fmt='k')
ydata = np.abs(datachunk['n_p scaled'])
binned_median_plot(xdata, ydata, nxbins = nbins,fighandle = fig, axhandle = axs[0,0], fmt='r')
ydata = np.abs(datachunk['n_p nocme'])
binned_median_plot(xdata, ydata, nxbins = nbins,fighandle = fig, axhandle = axs[0,0], fmt='b')
axs[0,0].set_ylabel(ylab)
axs[0,0].set_ylim((ymin,ymax))
axs[0,0].text(0.03,0.9,'(a)', fontsize = 14, transform=axs[0,0].transAxes, backgroundcolor = 'w')
axs[0,0].legend([r'Observed', r'$R$-scaled','No ICMEs'])
axs[0,0].get_xaxis().set_ticklabels([])

xdata = (datachunk['pos_lat'])
ydata = np.abs(datachunk[param])
binned_median_plot(xdata, ydata, nxbins = nbins, fighandle = fig, axhandle = axs[0,1], fmt='k')
ydata = np.abs(datachunk['n_p scaled'])
binned_median_plot(xdata, ydata, nxbins = nbins,fighandle = fig, axhandle = axs[0,1], fmt='r')
ydata = np.abs(datachunk['n_p nocme'])
binned_median_plot(xdata, ydata, nxbins = nbins,fighandle = fig, axhandle = axs[0,1], fmt='b')
axs[0,1].set_ylim((ymin,ymax))
axs[0,1].text(0.03,0.9,'(b)', fontsize = 14, transform=axs[0,1].transAxes, backgroundcolor = 'w')
axs[0,1].get_yaxis().set_ticklabels([])
axs[0,1].get_xaxis().set_ticklabels([])

xdata = np.abs(datachunk['pos_lat'])
ydata = np.abs(datachunk[param])
binned_median_plot(xdata, ydata, nxbins = nbins, fighandle = fig, axhandle = axs[0,2], fmt='k')
ydata = np.abs(datachunk['n_p scaled'])
binned_median_plot(xdata, ydata, nxbins = nbins,fighandle = fig, axhandle = axs[0,2], fmt='r')
ydata = np.abs(datachunk['n_p nocme'])
binned_median_plot(xdata, ydata, nxbins = nbins,fighandle = fig, axhandle = axs[0,2], fmt='b')
axs[0,2].set_ylim((ymin,ymax))
axs[0,2].text(0.03,0.9,'(c)', fontsize = 14, transform=axs[0,2].transAxes, backgroundcolor = 'w')
axs[0,2].get_yaxis().set_ticklabels([])
axs[0,2].get_xaxis().set_ticklabels([])


param = 'Vr'
ymin = 375
ymax = 425
ylab = r'$V_r$ [km s$^{-1}$]'

xdata = np.abs(datachunk['pos_r'])
ydata = np.abs(datachunk[param])
binned_median_plot(xdata, ydata, nxbins = nbins,fighandle = fig, axhandle = axs[1,0], fmt='k')
ydata = np.abs(datachunk['Vr scaled'])
binned_median_plot(xdata, ydata, nxbins = nbins,fighandle = fig, axhandle = axs[1,0], fmt='r')
ydata = np.abs(datachunk['Vr nocme'])
binned_median_plot(xdata, ydata, nxbins = nbins,fighandle = fig, axhandle = axs[1,0], fmt='b')
axs[1,0].set_ylabel(ylab)
axs[1,0].set_ylim((ymin,ymax))
axs[1,0].text(0.03,0.9,'(d)', fontsize = 14, transform=axs[1,0].transAxes, backgroundcolor = 'w')
axs[1,0].get_xaxis().set_ticklabels([])

xdata = (datachunk['pos_lat'])
ydata = np.abs(datachunk[param])
binned_median_plot(xdata, ydata, nxbins = nbins, fighandle = fig, axhandle = axs[1,1], fmt='k')
ydata = np.abs(datachunk['Vr scaled'])
binned_median_plot(xdata, ydata, nxbins = nbins,fighandle = fig, axhandle = axs[1,1], fmt='r')
ydata = np.abs(datachunk['Vr nocme'])
binned_median_plot(xdata, ydata, nxbins = nbins,fighandle = fig, axhandle = axs[1,1], fmt='b')
axs[1,1].set_ylim((ymin,ymax))
axs[1,1].text(0.03,0.9,'(e)', fontsize = 14, transform=axs[1,1].transAxes, backgroundcolor = 'w')
axs[1,1].get_yaxis().set_ticklabels([])
axs[1,1].get_xaxis().set_ticklabels([])

xdata = np.abs(datachunk['pos_lat'])
ydata = np.abs(datachunk[param])
binned_median_plot(xdata, ydata, nxbins = nbins, fighandle = fig, axhandle = axs[1,2], fmt='k')
ydata = np.abs(datachunk['Vr scaled'])
binned_median_plot(xdata, ydata, nxbins = nbins,fighandle = fig, axhandle = axs[1,2], fmt='r')
ydata = np.abs(datachunk['Vr nocme'])
binned_median_plot(xdata, ydata, nxbins = nbins,fighandle = fig, axhandle = axs[1,2], fmt='b')
axs[1,2].set_ylim((ymin,ymax))
axs[1,2].text(0.03,0.9,'(f)', fontsize = 14, transform=axs[1,2].transAxes, backgroundcolor = 'w')
axs[1,2].get_yaxis().set_ticklabels([])
axs[1,2].get_xaxis().set_ticklabels([])

param = 'Bmag'
ymin = 4.6
ymax = 5.4
ylab = r'$|$B$|$ [nT]'

xdata = np.abs(datachunk['pos_r'])
ydata = np.abs(datachunk[param])
binned_median_plot(xdata, ydata, nxbins = nbins,fighandle = fig, axhandle = axs[2,0], fmt='k')
ydata = np.abs(datachunk['Bmag scaled'])
binned_median_plot(xdata, ydata, nxbins = nbins,fighandle = fig, axhandle = axs[2,0], fmt='r')
ydata = np.abs(datachunk['Bmag nocme'])
binned_median_plot(xdata, ydata, nxbins = nbins,fighandle = fig, axhandle = axs[2,0], fmt='b')
axs[2,0].set_ylabel(ylab)
axs[2,0].set_ylim((ymin,ymax))
axs[2,0].set_xlabel(r'R [$r_\odot$]')
axs[2,0].text(0.03,0.9,'(g)', fontsize = 14, transform=axs[2,0].transAxes, backgroundcolor = 'w')

xdata = (datachunk['pos_lat'])
ydata = np.abs(datachunk[param])
binned_median_plot(xdata, ydata, nxbins = nbins, fighandle = fig, axhandle = axs[2,1], fmt='k')
ydata = np.abs(datachunk['Bmag scaled'])
binned_median_plot(xdata, ydata, nxbins = nbins,fighandle = fig, axhandle = axs[2,1], fmt='r')
ydata = np.abs(datachunk['Bmag nocme'])
binned_median_plot(xdata, ydata, nxbins = nbins,fighandle = fig, axhandle = axs[2,1], fmt='b')
axs[2,1].set_ylim((ymin,ymax))
axs[2,1].set_xlabel(r'$\theta$ [deg]')
axs[2,1].text(0.03,0.9,'(h)', fontsize = 14, transform=axs[2,1].transAxes, backgroundcolor = 'w')
axs[2,1].get_yaxis().set_ticklabels([])

xdata = np.abs(datachunk['pos_lat'])
ydata = np.abs(datachunk[param])
binned_median_plot(xdata, ydata, nxbins = nbins, fighandle = fig, axhandle = axs[2,2], fmt='k')
ydata = np.abs(datachunk['Bmag scaled'])
binned_median_plot(xdata, ydata, nxbins = nbins,fighandle = fig, axhandle = axs[2,2], fmt='r')
ydata = np.abs(datachunk['Bmag nocme'])
binned_median_plot(xdata, ydata, nxbins = nbins,fighandle = fig, axhandle = axs[2,2], fmt='b')
axs[2,2].set_ylim((ymin,ymax))
axs[2,2].set_xlabel(r'$|\theta|$ [deg]')
axs[2,2].text(0.03,0.9,'(i)', fontsize = 14, transform=axs[2,2].transAxes, backgroundcolor = 'w')
axs[2,2].get_yaxis().set_ticklabels([])

for axlist in axs:
    for ax in axlist:
        ax.minorticks_on()
        
fig.tight_layout(rect=[0, 0.1, 1, 0.95])
fig.savefig( os.path.join(fig_dir,'orbitalparams_medians_summary.pdf')) 

## <codecell> solar wind speed variation with frac year





# nbins = nbins

# fig, axs = plt.subplots(nrows = 3, ncols = 3, figsize=(10, 10))

# mask = (ace1hr['ssn'] <= 30000)
# datachunk = ace1hr.loc[mask] 

# xdata = np.abs(datachunk['frac_of_yr'])
# ydata = np.abs(datachunk['n_p'])
# contour2d(xdata, ydata, nxbins = nbins, nybins = nbins, fighandle = fig, axhandle = axs[0,0],
#                   xmin = 0, xmax = 1, ymin = nmin, ymax = 3*nmax/4,  plotcbar = False)
# axs[0,0].set_title('All data')
# axs[0,0].set_ylabel(r'$n_{P}$ [cm$^{-3}$]')
# axs[0,0].text(0.03,0.9,'(a)', fontsize = 14, transform=axs[0,0].transAxes, backgroundcolor = 'w')


# ydata = np.abs(datachunk['Vr'])
# contour2d(xdata, ydata, nxbins = nbins, nybins = nbins, fighandle = fig, axhandle = axs[1,0],
#                  xmin = 0, xmax = 1, ymin = vmin, ymax = (vmax-vmin)/2 +vmin,  plotcbar = False)
# axs[1,0].set_ylabel(r'$V_{r}$ [km s$^{-1}$]')
# axs[1,0].text(0.03,0.9,'(d)', fontsize = 14, transform=axs[1,0].transAxes, backgroundcolor = 'w')


# ydata = np.abs(datachunk['Bmag'])
# contour2d(xdata, ydata, nxbins = nbins, nybins = nbins, fighandle = fig, axhandle = axs[2,0],
#                 xmin = 0, xmax = 1,  ymin = bmin, ymax = bmax/2,  plotcbar = False)
# axs[2,0].set_ylabel(r'$|$B$|$ [nT]')
# axs[2,0].set_xlabel(r'Fraction of year')
# axs[2,0].text(0.03,0.9,'(g)', fontsize = 14, transform=axs[2,0].transAxes, backgroundcolor = 'w')





# mask = ace1hr['ssn'] <= ssn_thresh
# datachunk = ace1hr.loc[mask] 

# xdata = np.abs(datachunk['frac_of_yr'])
# ydata = np.abs(datachunk['n_p'])
# contour2d(xdata, ydata, nxbins = nbins, nybins = nbins, fighandle = fig, axhandle = axs[0,1],
#                   xmin = 0, xmax = 1, ymin = nmin, ymax = 3*nmax/4,  plotcbar = False)
# axs[0,1].set_title('SSN <= ' +str(ssn_thresh))
# axs[0,1].set_ylabel(r'$n_{P}$ [cm$^{-3}$]')
# axs[0,1].text(0.03,0.9,'(b)', fontsize = 14, transform=axs[0,1].transAxes, backgroundcolor = 'w')

# ydata = np.abs(datachunk['Vr'])
# contour2d(xdata, ydata, nxbins = nbins, nybins = nbins, fighandle = fig, axhandle = axs[1,1],
#                  xmin = 0, xmax = 1, ymin = vmin, ymax = (vmax-vmin)/2 +vmin,  plotcbar = False)
# axs[1,1].set_ylabel(r'$V_{r}$ [km s$^{-1}$]')
# axs[1,1].text(0.03,0.9,'(e)', fontsize = 14, transform=axs[1,1].transAxes, backgroundcolor = 'w')


# ydata = np.abs(datachunk['Bmag'])
# contour2d(xdata, ydata, nxbins = nbins, nybins = nbins, fighandle = fig, axhandle = axs[2,1],
#                 xmin = 0, xmax = 1,  ymin = bmin, ymax = bmax/2,  plotcbar = False)
# axs[2,1].set_ylabel(r'$|$B$|$ [nT]')
# axs[2,1].set_xlabel(r'Fraction of year')
# axs[2,1].text(0.03,0.9,'(h)', fontsize = 14, transform=axs[2,1].transAxes, backgroundcolor = 'w')




# mask = (ace1hr['ssn'] > ssn_thresh)
# datachunk = ace1hr.loc[mask] 

# xdata = np.abs(datachunk['frac_of_yr'])
# ydata = np.abs(datachunk['n_p'])
# contour2d(xdata, ydata, nxbins = nbins, nybins = nbins, fighandle = fig, axhandle = axs[0,2],
#                   xmin = 0, xmax = 1, ymin = nmin, ymax = 3*nmax/4,  plotcbar = False)
# axs[0,2].set_title('SSN > ' +str(ssn_thresh))
# axs[0,2].set_ylabel(r'$n_{P}$ [cm$^{-3}$]')
# axs[0,2].text(0.03,0.9,'(c)', fontsize = 14, transform=axs[0,2].transAxes, backgroundcolor = 'w')

# ydata = np.abs(datachunk['Vr'])
# contour2d(xdata, ydata, nxbins = nbins, nybins = nbins, fighandle = fig, axhandle = axs[1,2],
#                  xmin = 0, xmax = 1, ymin = vmin, ymax = (vmax-vmin)/2 +vmin,  plotcbar = False)
# axs[1,2].set_ylabel(r'$V_{r}$ [km s$^{-1}$]')
# axs[1,2].text(0.03,0.9,'(f)', fontsize = 14, transform=axs[1,2].transAxes, backgroundcolor = 'w')


# ydata = np.abs(datachunk['Bmag'])
# fig, ax, cmap = contour2d(xdata, ydata, nxbins = nbins, nybins = nbins, fighandle = fig, axhandle = axs[2,2],
#                 xmin = 0, xmax = 1,  ymin = bmin, ymax = bmax/2,  plotcbar = False)
# axs[2,2].set_ylabel(r'$|$B$|$ [nT]')
# axs[2,2].set_xlabel(r'Fraction of year')
# axs[2,2].text(0.03,0.9,'(i)', fontsize = 14, transform=axs[2,2].transAxes, backgroundcolor = 'w')

# fig.tight_layout(rect=[0, 0.1, 1, 0.95])
# cax = plt.axes([0.12, 0.06, 0.8, 0.02]) #Left,bottom, length, width
# clb=plt.colorbar(cmap, cax=cax,orientation="horizontal")
# clb.set_label('log(counts)',fontsize=14)

# fig.savefig(fig_dir + 'solarwind_fracyr_SPE.pdf')  


## <codecell> solar wind speed variation with frac year - box plot

# nbins = nbins

# fig, axs = plt.subplots(nrows = 3, ncols = 3, figsize=(10, 10))

# mask = (ace1hr['ssn'] <= 30000)
# datachunk = ace1hr.loc[mask] 

# xdata = np.abs(datachunk['frac_of_yr'])
# ydata = np.abs(datachunk['n_p'])
# binned_box_plots(xdata, ydata,  fighandle = fig, axhandle = axs[0,0], xmin=0, xmax=1)
# axs[0,0].set_title('All data')
# axs[0,0].set_ylabel(r'$n_{P}$ [cm$^{-3}$]')
# axs[0,0].text(0.03,0.9,'(a)', fontsize = 14, transform=axs[0,0].transAxes, backgroundcolor = 'w')


# ydata = np.abs(datachunk['Vr'])
# binned_box_plots(xdata, ydata, nxbins = nbins, fighandle = fig, axhandle = axs[1,0],
#                  xmin = 0, xmax = 1)
# axs[1,0].set_ylabel(r'$V_{r}$ [km s$^{-1}$]')
# axs[1,0].text(0.03,0.9,'(d)', fontsize = 14, transform=axs[1,0].transAxes, backgroundcolor = 'w')


# ydata = np.abs(datachunk['Bmag'])
# binned_box_plots(xdata, ydata, nxbins = nbins,  fighandle = fig, axhandle = axs[2,0],
#                 xmin = 0, xmax = 1)
# axs[2,0].set_ylabel(r'$|$B$|$ [nT]')
# axs[2,0].set_xlabel(r'Fraction of year')
# axs[2,0].text(0.03,0.9,'(g)', fontsize = 14, transform=axs[2,0].transAxes, backgroundcolor = 'w')





# mask = ace1hr['ssn'] <= ssn_thresh
# datachunk = ace1hr.loc[mask] 

# xdata = np.abs(datachunk['frac_of_yr'])
# ydata = np.abs(datachunk['n_p'])
# binned_box_plots(xdata, ydata, nxbins = nbins,  fighandle = fig, axhandle = axs[0,1],
#                   xmin = 0, xmax = 1)
# axs[0,1].set_title('SSN <= ' +str(ssn_thresh))
# axs[0,1].set_ylabel(r'$n_{P}$ [cm$^{-3}$]')
# axs[0,1].text(0.03,0.9,'(b)', fontsize = 14, transform=axs[0,1].transAxes, backgroundcolor = 'w')

# ydata = np.abs(datachunk['Vr'])
# binned_box_plots(xdata, ydata, nxbins = nbins, fighandle = fig, axhandle = axs[1,1],
#                  xmin = 0, xmax = 1)
# axs[1,1].set_ylabel(r'$V_{r}$ [km s$^{-1}$]')
# axs[1,1].text(0.03,0.9,'(e)', fontsize = 14, transform=axs[1,1].transAxes, backgroundcolor = 'w')


# ydata = np.abs(datachunk['Bmag'])
# binned_box_plots(xdata, ydata, nxbins = nbins,  fighandle = fig, axhandle = axs[2,1],
#                 xmin = 0, xmax = 1)
# axs[2,1].set_ylabel(r'$|$B$|$ [nT]')
# axs[2,1].set_xlabel(r'Fraction of year')
# axs[2,1].text(0.03,0.9,'(h)', fontsize = 14, transform=axs[2,1].transAxes, backgroundcolor = 'w')




# mask = (ace1hr['ssn'] > ssn_thresh)
# datachunk = ace1hr.loc[mask] 

# xdata = np.abs(datachunk['frac_of_yr'])
# ydata = np.abs(datachunk['n_p'])
# binned_box_plots(xdata, ydata, nxbins = nbins, fighandle = fig, axhandle = axs[0,2],
#                   xmin = 0, xmax = 1)
# axs[0,2].set_title('SSN > ' +str(ssn_thresh))
# axs[0,2].set_ylabel(r'$n_{P}$ [cm$^{-3}$]')
# axs[0,2].text(0.03,0.9,'(c)', fontsize = 14, transform=axs[0,2].transAxes, backgroundcolor = 'w')

# ydata = np.abs(datachunk['Vr'])
# binned_box_plots(xdata, ydata, nxbins = nbins,  fighandle = fig, axhandle = axs[1,2],
#                  xmin = 0, xmax = 1)
# axs[1,2].set_ylabel(r'$V_{r}$ [km s$^{-1}$]')
# axs[1,2].text(0.03,0.9,'(f)', fontsize = 14, transform=axs[1,2].transAxes, backgroundcolor = 'w')


# ydata = np.abs(datachunk['Bmag'])
# binned_box_plots(xdata, ydata, nxbins = nbins,  fighandle = fig, axhandle = axs[2,2],
#                 xmin = 0, xmax = 1)
# axs[2,2].set_ylabel(r'$|$B$|$ [nT]')
# axs[2,2].set_xlabel(r'Fraction of year')
# axs[2,2].text(0.03,0.9,'(i)', fontsize = 14, transform=axs[2,2].transAxes, backgroundcolor = 'w')

# fig.tight_layout()
# fig.savefig(fig_dir + 'solarwind_fracyr_boxplots.pdf')  


# <codecell> solar wind param variation with frac year - medians

nplotmin = 3.3
nplotmax = 5.8
vplotmin = 368
vplotmax = 425
bplotmin = 3.9
bplotmax = 6.3

fig, axs = plt.subplots(nrows = 3, ncols = 3, figsize=(10, 10))

mask = (ace1hr['sai'] <= 1)
datachunk = ace1hr.loc[mask] 


xdata = np.abs(datachunk['frac_of_yr'])
ydata = np.abs(datachunk['n_p'])
binned_median_plot(xdata, ydata,  nxbins = nbins,  fighandle = fig, axhandle = axs[0,0], 
                   xmin=0, xmax=1, fmt='k')
ydata = np.abs(datachunk['n_p scaled'])
binned_median_plot(xdata, ydata,  nxbins = nbins,  fighandle = fig, axhandle = axs[0,0], 
                   xmin=0, xmax=1, fmt='r')
ydata = np.abs(datachunk['n_p nocme'])
binned_median_plot(xdata, ydata,  nxbins = nbins,  fighandle = fig, axhandle = axs[0,0], 
                   xmin=0, xmax=1, fmt='b')
axs[0,0].set_ylim((nplotmin,nplotmax))
axs[0,0].set_title('All data')
axs[0,0].set_ylabel(r'$n_{P}$ [cm$^{-3}$]')
axs[0,0].text(0.03,0.9,'(a)', fontsize = 14, transform=axs[0,0].transAxes, backgroundcolor = 'w')
axs[0,0].get_xaxis().set_ticklabels([])
yy = axs[0,0].get_ylim(); axs[0,0].plot([0.5, 0.5] ,yy ,'k')


ydata = np.abs(datachunk['Vr'])
binned_median_plot(xdata, ydata, nxbins = nbins, fighandle = fig, axhandle = axs[1,0],
                 xmin = 0, xmax = 1, fmt='k')
ydata = np.abs(datachunk['Vr scaled'])
binned_median_plot(xdata, ydata,  nxbins = nbins,  fighandle = fig, axhandle = axs[1,0], 
                   xmin=0, xmax=1, fmt='r')
ydata = np.abs(datachunk['Vr nocme'])
binned_median_plot(xdata, ydata,  nxbins = nbins,  fighandle = fig, axhandle = axs[1,0], 
                   xmin=0, xmax=1, fmt='b')
axs[1,0].set_ylim((vplotmin,vplotmax))
axs[1,0].set_ylabel(r'$V_{r}$ [km s$^{-1}$]')
axs[1,0].text(0.03,0.9,'(d)', fontsize = 14, transform=axs[1,0].transAxes, backgroundcolor = 'w')
axs[1,0].get_xaxis().set_ticklabels([])
yy = axs[1,0].get_ylim(); axs[1,0].plot([0.5, 0.5] ,yy ,'k')


ydata = np.abs(datachunk['Bmag'])
binned_median_plot(xdata, ydata, nxbins = nbins,  fighandle = fig, axhandle = axs[2,0],
                xmin = 0, xmax = 1, fmt='k')
ydata = np.abs(datachunk['Bmag scaled'])
binned_median_plot(xdata, ydata, nxbins = nbins,  fighandle = fig, axhandle = axs[2,0],
                xmin = 0, xmax = 1, fmt='r')
ydata = np.abs(datachunk['Bmag nocme'])
binned_median_plot(xdata, ydata, nxbins = nbins,  fighandle = fig, axhandle = axs[2,0],
                xmin = 0, xmax = 1, fmt='b')
axs[2,0].set_ylim((bplotmin,bplotmax))
axs[2,0].set_ylabel(r'$|$B$|$ [nT]')
axs[2,0].set_xlabel(r'Fraction of year, $F$')
axs[2,0].text(0.03,0.9,'(g)', fontsize = 14, transform=axs[2,0].transAxes, backgroundcolor = 'w')
yy = axs[2,0].get_ylim(); axs[2,0].plot([0.5, 0.5] ,yy ,'k')




mask = ace1hr['sai'] <= sai_thresh_low
datachunk = ace1hr.loc[mask] 

xdata = np.abs(datachunk['frac_of_yr'])
ydata = np.abs(datachunk['n_p'])
binned_median_plot(xdata, ydata, nxbins = nbins,  fighandle = fig, axhandle = axs[0,1],
                  xmin = 0, xmax = 1, fmt='k')
ydata = np.abs(datachunk['n_p scaled'])
binned_median_plot(xdata, ydata, nxbins = nbins,  fighandle = fig, axhandle = axs[0,1],
                   xmin=0, xmax=1, fmt='r')
ydata = np.abs(datachunk['n_p nocme'])
binned_median_plot(xdata, ydata, nxbins = nbins,  fighandle = fig, axhandle = axs[0,1],
                   xmin=0, xmax=1, fmt='b')
axs[0,1].set_ylim((nplotmin,nplotmax))
axs[0,1].set_title('Solar min')
axs[0,1].text(0.03,0.9,'(b)', fontsize = 14, transform=axs[0,1].transAxes, backgroundcolor = 'w')
axs[0,1].legend([r'Observed', r'$R$-scaled', 'No ICMEs'])
axs[0,1].get_yaxis().set_ticklabels([])
axs[0,1].get_xaxis().set_ticklabels([])
yy = axs[0,1].get_ylim(); axs[0,1].plot([0.5, 0.5] ,yy ,'k')

ydata = np.abs(datachunk['Vr'])
binned_median_plot(xdata, ydata, nxbins = nbins, fighandle = fig, axhandle = axs[1,1],
                 xmin = 0, xmax = 1, fmt='k')
ydata = np.abs(datachunk['Vr scaled'])
binned_median_plot(xdata, ydata, nxbins = nbins, fighandle = fig, axhandle = axs[1,1],
                 xmin = 0, xmax = 1, fmt='r')
ydata = np.abs(datachunk['Vr nocme'])
binned_median_plot(xdata, ydata, nxbins = nbins, fighandle = fig, axhandle = axs[1,1],
                 xmin = 0, xmax = 1, fmt='b')
axs[1,1].set_ylim((vplotmin,vplotmax))
axs[1,1].text(0.03,0.9,'(e)', fontsize = 14, transform=axs[1,1].transAxes, backgroundcolor = 'w')
axs[1,1].get_yaxis().set_ticklabels([])
axs[1,1].get_xaxis().set_ticklabels([])
yy = axs[1,1].get_ylim(); axs[1,1].plot([0.5, 0.5] ,yy ,'k')

ydata = np.abs(datachunk['Bmag'])
binned_median_plot(xdata, ydata, nxbins = nbins,  fighandle = fig, axhandle = axs[2,1],
                xmin = 0, xmax = 1, fmt='k')
ydata = np.abs(datachunk['Bmag scaled'])
binned_median_plot(xdata, ydata, nxbins = nbins,  fighandle = fig, axhandle = axs[2,1],
                xmin = 0, xmax = 1, fmt='r')
ydata = np.abs(datachunk['Bmag nocme'])
binned_median_plot(xdata, ydata, nxbins = nbins,  fighandle = fig, axhandle = axs[2,1],
                xmin = 0, xmax = 1, fmt='b')
axs[2,1].set_ylim((bplotmin,bplotmax))
axs[2,1].set_xlabel(r'Fraction of year, $F$')
axs[2,1].text(0.03,0.9,'(h)', fontsize = 14, transform=axs[2,1].transAxes, backgroundcolor = 'w')
axs[2,1].get_yaxis().set_ticklabels([])
yy = axs[2,1].get_ylim(); axs[2,1].plot([0.5, 0.5] ,yy ,'k')




mask = (ace1hr['sai'] >= sai_thresh_high)
datachunk = ace1hr.loc[mask] 

xdata = np.abs(datachunk['frac_of_yr'])
ydata = np.abs(datachunk['n_p'])
binned_median_plot(xdata, ydata, nxbins = nbins, fighandle = fig, axhandle = axs[0,2],
                  xmin = 0, xmax = 1, fmt='k')
ydata = np.abs(datachunk['n_p scaled'])
binned_median_plot(xdata, ydata, nxbins = nbins,   fighandle = fig, axhandle = axs[0,2], 
                   xmin=0, xmax=1, fmt='r')
ydata = np.abs(datachunk['n_p nocme'])
binned_median_plot(xdata, ydata, nxbins = nbins,   fighandle = fig, axhandle = axs[0,2], 
                   xmin=0, xmax=1, fmt='b')
axs[0,2].set_ylim((nplotmin,nplotmax))
axs[0,2].set_title('Solar max')
axs[0,2].text(0.03,0.9,'(c)', fontsize = 14, transform=axs[0,2].transAxes, backgroundcolor = 'w')
axs[0,2].get_yaxis().set_ticklabels([])
axs[0,2].get_xaxis().set_ticklabels([])
yy = axs[0,2].get_ylim(); axs[0,2].plot([0.5, 0.5] ,yy ,'k')

ydata = np.abs(datachunk['Vr'])
binned_median_plot(xdata, ydata, nxbins = nbins,  fighandle = fig, axhandle = axs[1,2],
                 xmin = 0, xmax = 1, fmt='k')
ydata = np.abs(datachunk['Vr scaled'])
binned_median_plot(xdata, ydata, nxbins = nbins,  fighandle = fig, axhandle = axs[1,2],
                 xmin = 0, xmax = 1, fmt='r')
ydata = np.abs(datachunk['Vr nocme'])
binned_median_plot(xdata, ydata, nxbins = nbins,  fighandle = fig, axhandle = axs[1,2],
                 xmin = 0, xmax = 1, fmt='b')
axs[1,2].set_ylim((vplotmin,vplotmax))
axs[1,2].text(0.03,0.9,'(f)', fontsize = 14, transform=axs[1,2].transAxes, backgroundcolor = 'w')
axs[1,2].get_yaxis().set_ticklabels([])
axs[1,2].get_xaxis().set_ticklabels([])
yy = axs[1,2].get_ylim(); axs[1,2].plot([0.5, 0.5] ,yy ,'k')

ydata = np.abs(datachunk['Bmag'])
binned_median_plot(xdata, ydata, nxbins = nbins,  fighandle = fig, axhandle = axs[2,2],
                xmin = 0, xmax = 1, fmt='k')
ydata = np.abs(datachunk['Bmag scaled'])
binned_median_plot(xdata, ydata, nxbins = nbins,  fighandle = fig, axhandle = axs[2,2],
                xmin = 0, xmax = 1, fmt='r')
ydata = np.abs(datachunk['Bmag nocme'])
binned_median_plot(xdata, ydata, nxbins = nbins,  fighandle = fig, axhandle = axs[2,2],
                xmin = 0, xmax = 1, fmt='b')
axs[2,2].set_ylim((bplotmin,bplotmax))
axs[2,2].set_xlabel(r'Fraction of year, $F$')
axs[2,2].text(0.03,0.9,'(i)', fontsize = 14, transform=axs[2,2].transAxes, backgroundcolor = 'w')
axs[2,2].get_yaxis().set_ticklabels([])
yy = axs[2,2].get_ylim(); axs[2,2].plot([0.5, 0.5] ,yy ,'k')

fig.tight_layout()
fig.savefig( os.path.join(fig_dir, 'solarwind_fracyr_medians.pdf'))  

## <codecell> solar wind pressure
# nbins = nbins
# pmax = 4





# fig, axs = plt.subplots(nrows = 1, ncols = 3, figsize=(10, 4))

# mask = (ace1hr['sai'] <= 30000)
# datachunk = ace1hr.loc[mask] 

# xdata = np.abs(datachunk['frac_of_yr'])
# ydata = datachunk['Pdyn'] + datachunk['P_B']
# contour2d(xdata, ydata, nxbins = nbins, nybins = nbins, fighandle = fig, axhandle = axs[0],
#                   xmin = 0, xmax = 1, ymin = 0, ymax = pmax,  plotcbar = False)
# axs[0].set_title('All data')
# axs[0].set_ylabel(r'$P_{DYN}$ [nPa]')
# axs[0].set_xlabel(r'Fraction of year')
# axs[0].text(0.03,0.9,'(a)', fontsize = 14, transform=axs[0].transAxes, backgroundcolor = 'w')

# mask = (ace1hr['sai'] <= sai_thresh_low)
# datachunk = ace1hr.loc[mask] 

# xdata = np.abs(datachunk['frac_of_yr'])
# ydata = datachunk['Pdyn'] + datachunk['P_B']
# contour2d(xdata, ydata, nxbins = nbins, nybins = nbins, fighandle = fig, axhandle = axs[1],
#                   xmin = 0, xmax = 1, ymin = 0, ymax = pmax,  plotcbar = False)
# axs[1].set_title('Solar min')
# axs[1].set_ylabel(r'$P_{DYN}$ [nPa]')
# axs[1].set_xlabel(r'Fraction of year')
# axs[1].text(0.03,0.9,'(b)', fontsize = 14, transform=axs[1].transAxes, backgroundcolor = 'w')

# mask = (ace1hr['sai'] > sai_thresh_high)
# datachunk = ace1hr.loc[mask] 

# xdata = np.abs(datachunk['frac_of_yr'])
# ydata = datachunk['Pdyn'] + datachunk['P_B']
# contour2d(xdata, ydata, nxbins = nbins, nybins = nbins, fighandle = fig, axhandle = axs[2],
#                   xmin = 0, xmax = 1, ymin = 0, ymax = pmax,  plotcbar = False)
# axs[2].set_title('Solar max')
# axs[2].set_ylabel(r'$P_{DYN}$ [nPa]')
# axs[2].set_xlabel(r'Fraction of year')
# axs[2].text(0.03,0.9,'(c)', fontsize = 14, transform=axs[2].transAxes, backgroundcolor = 'w')

# fig.tight_layout(rect=[0, 0, 0.9, 1])
# cax = plt.axes([0.9, 0.18, 0.02, 0.72]) #Left,bottom, length, width
# clb=plt.colorbar(cmap, cax=cax,orientation="vertical")
# clb.set_label('log(counts)',fontsize=14)

# fig.savefig(fig_dir + 'pdyn_fracyr_SPE.pdf')  


# # <codecell> alpha-to-proton ratio realtion to other solar wind parameters

# fig, axs = plt.subplots(nrows = 3, ncols = 1, figsize=(6, 10))

# xdata = np.abs(datachunk['n_p'])
# ydata = np.abs(datachunk['ratio_a2p'])
# binned_median_plot(xdata, ydata, nxbins = nbins,  fighandle = fig, axhandle = axs[0],
#                  xmin = nmin, xmax = nmax, )
# axs[0].set_xlabel(r'$n_P$ [cm$^{-3}$]')
# axs[0].set_ylabel(r'$\alpha:p$ [cm$^{-3}$]')
# axs[0].text(0.02,0.9,'(a)', fontsize = 14, transform=axs[0].transAxes, backgroundcolor = 'w')

# xdata = np.abs(datachunk['Vr'])
# ydata = np.abs(datachunk['ratio_a2p'])
# binned_median_plot(xdata, ydata, nxbins = nbins,  fighandle = fig, axhandle = axs[1],
#                  xmin = vmin, xmax = vmax, )
# axs[1].set_xlabel(r'$V_{SW}$ [km s$^{-1}$]')
# axs[1].set_ylabel(r'$\alpha:p$ [cm$^{-3}$]')
# axs[1].text(0.02,0.9,'(b)', fontsize = 14, transform=axs[1].transAxes, backgroundcolor = 'w')

# xdata = np.abs(datachunk['Bmag'])
# ydata = np.abs(datachunk['ratio_a2p'])
# binned_median_plot(xdata, ydata, nxbins = nbins, fighandle = fig, axhandle = axs[2],
#                  xmin = bmin, xmax = bmax, )
# axs[2].set_xlabel(r'$|B|$ [nT]')
# axs[2].set_ylabel(r'$\alpha:p$ [cm$^{-3}$]')
# axs[2].text(0.02,0.9,'(c)', fontsize = 14, transform=axs[2].transAxes, backgroundcolor = 'w')

# fig.set_tight_layout(True)
# fig.savefig(fig_dir + 'a_to_p_interdependence_medians.pdf') 


# <codecell> ram pressure and coupling function

fig, axs = plt.subplots(nrows =2, ncols = 3, figsize=(10, 8))


pplotmin = 1.15
pplotmax = 2
powplotmin = 0.00215
powplotmax = 0.00365


# mask = (ace1hr['ssn'] <= 30000)
# datachunk = ace1hr.loc[mask] 

# xdata = np.abs(datachunk['frac_of_yr'])
# ydata = datachunk['ratio_a2p'] 
# binned_median_plot(xdata, ydata, nxbins = nbins,fighandle = fig, axhandle = axs[0,0],
#                   xmin = 0, xmax = 1)
# axs[0,0].set_title('All data')
# axs[0,0].set_ylabel(r'$\alpha$:p')
# axs[0,0].set_xlabel(r'Fraction of year')
# axs[0,0].text(0.03,0.9,'(a)', fontsize = 14, transform=axs[0,0].transAxes, backgroundcolor = 'w')

# mask = (ace1hr['ssn'] <= ssn_thresh)
# datachunk = ace1hr.loc[mask] 

# xdata = np.abs(datachunk['frac_of_yr'])
# ydata = datachunk['ratio_a2p'] 
# binned_median_plot(xdata, ydata, nxbins = nbins,fighandle = fig, axhandle = axs[0,1],
#                   xmin = 0, xmax = 1,)
# axs[0,1].set_title('SSN <= ' + str(ssn_thresh))
# axs[0,1].set_xlabel(r'Fraction of year')
# axs[0,1].text(0.03,0.9,'(b)', fontsize = 14, transform=axs[0,1].transAxes, backgroundcolor = 'w')

# mask = (ace1hr['ssn'] > ssn_thresh)
# datachunk = ace1hr.loc[mask] 

# xdata = np.abs(datachunk['frac_of_yr'])
# ydata = datachunk['ratio_a2p'] 
# binned_median_plot(xdata, ydata, nxbins = nbins, fighandle = fig, axhandle = axs[0,2],
#                   xmin = 0, xmax = 1,)
# axs[0,2].set_title('SSN > ' + str(ssn_thresh))
# axs[0,2].set_xlabel(r'Fraction of year')
# axs[0,2].text(0.03,0.9,'(c)', fontsize = 14, transform=axs[0,2].transAxes, backgroundcolor = 'w')


mask = (ace1hr['sai'] <= 1)
datachunk = ace1hr.loc[mask] 
row = 0; column = 0

xdata = np.abs(datachunk['frac_of_yr'])
ydata = datachunk['Pdyn'] #+ datachunk['P_B']
binned_median_plot(xdata, ydata, nxbins = nbins,fighandle = fig, axhandle = axs[row,column],
                  xmin = 0, xmax = 1, fmt='k')
ydata = datachunk['Pdyn scaled'] #+ datachunk['P_B']
binned_median_plot(xdata, ydata, nxbins = nbins,fighandle = fig, axhandle = axs[row,column],
                  xmin = 0, xmax = 1, fmt='r')
ydata = datachunk['Pdyn nocme'] #+ datachunk['P_B']
binned_median_plot(xdata, ydata, nxbins = nbins,fighandle = fig, axhandle = axs[row,column],
                  xmin = 0, xmax = 1, fmt='b')
axs[row,column].set_ylim((pplotmin,pplotmax))
axs[row,column].set_ylabel(r'$P_{DYN}$ [nPa]')
axs[row,column].set_title('All data')
axs[row,column].text(0.03,0.9,'(a)', fontsize = 14, transform=axs[row,column].transAxes, backgroundcolor = 'w')
axs[row,column].get_xaxis().set_ticklabels([])
yy = axs[row,column].get_ylim(); axs[row,column].plot([0.5, 0.5] ,yy ,'k')


mask = (ace1hr['sai'] <= sai_thresh_low)
datachunk = ace1hr.loc[mask] 
row = 0; column = 1

xdata = np.abs(datachunk['frac_of_yr'])
ydata = datachunk['Pdyn'] #+ datachunk['P_B']
binned_median_plot(xdata, ydata, nxbins = nbins,fighandle = fig, axhandle = axs[row,column],
                  xmin = 0, xmax = 1, fmt='k')
ydata = datachunk['Pdyn scaled'] #+ datachunk['P_B']
binned_median_plot(xdata, ydata, nxbins = nbins,fighandle = fig, axhandle = axs[row,column],
                  xmin = 0, xmax = 1, fmt='r')
ydata = datachunk['Pdyn nocme'] #+ datachunk['P_B']
binned_median_plot(xdata, ydata, nxbins = nbins,fighandle = fig, axhandle = axs[row,column],
                  xmin = 0, xmax = 1, fmt='b')
axs[row,column].set_ylim((pplotmin,pplotmax))
axs[row,column].set_title('Solar min')
axs[row,column].text(0.03,0.9,'(b)', fontsize = 14, transform=axs[row,column].transAxes, backgroundcolor = 'w')
axs[row,column].legend([r'Observed', r'$R$-scaled','No ICMEs'], loc = 'upper right')
axs[row,column].get_yaxis().set_ticklabels([])
axs[row,column].get_xaxis().set_ticklabels([])
yy = axs[row,column].get_ylim(); axs[row,column].plot([0.5, 0.5] ,yy ,'k')

mask = (ace1hr['sai'] >= sai_thresh_high)
datachunk = ace1hr.loc[mask] 
row = 0; column = 2

xdata = np.abs(datachunk['frac_of_yr'])
ydata = datachunk['Pdyn'] #+ datachunk['P_B']
binned_median_plot(xdata, ydata, nxbins = nbins,fighandle = fig, axhandle = axs[row,column],
                  xmin = 0, xmax = 1, fmt='k')
ydata = datachunk['Pdyn scaled'] #+ datachunk['P_B']
binned_median_plot(xdata, ydata, nxbins = nbins,fighandle = fig, axhandle = axs[row,column],
                  xmin = 0, xmax = 1, fmt='r')
ydata = datachunk['Pdyn nocme'] #+ datachunk['P_B']
binned_median_plot(xdata, ydata, nxbins = nbins,fighandle = fig, axhandle = axs[row,column],
                  xmin = 0, xmax = 1, fmt='b')
axs[row,column].set_ylim((pplotmin,pplotmax))
axs[row,column].set_title('Solar max')
axs[row,column].text(0.03,0.9,'(c)', fontsize = 14, transform=axs[row,column].transAxes, backgroundcolor = 'w')
axs[row,column].get_yaxis().set_ticklabels([])
axs[row,column].get_xaxis().set_ticklabels([])
yy = axs[row,column].get_ylim(); axs[row,column].plot([0.5, 0.5] ,yy ,'k')




mask = (ace1hr['sai'] <= 1)
datachunk = ace1hr.loc[mask] 
row = 1; column = 0

xdata = np.abs(datachunk['frac_of_yr'])
ydata = datachunk['Pinput'] #+ datachunk['P_B']
binned_median_plot(xdata, ydata, nxbins = nbins,fighandle = fig, axhandle = axs[row,column],
                  xmin = 0, xmax = 1, fmt='k')
ydata = datachunk['Pinput scaled'] #+ datachunk['P_B']
binned_median_plot(xdata, ydata, nxbins = nbins,fighandle = fig, axhandle = axs[row,column],
                  xmin = 0, xmax = 1, fmt='r')
ydata = datachunk['Pinput nocme'] #+ datachunk['P_B']
binned_median_plot(xdata, ydata, nxbins = nbins,fighandle = fig, axhandle = axs[row,column],
                  xmin = 0, xmax = 1, fmt='b')
axs[row,column].set_ylim((powplotmin,powplotmax))
axs[row,column].set_ylabel(r'$P_{\alpha}$ [arbitrary units]')
axs[row,column].set_xlabel(r'Fraction of year')
axs[row,column].text(0.03,0.9,'(d)', fontsize = 14, transform=axs[row,column].transAxes, backgroundcolor = 'w')
#axs[row,column].legend([r'Observed', r'$R$-scaled'], loc = 'lower left')
yy = axs[row,column].get_ylim(); axs[row,column].plot([0.5, 0.5] ,yy ,'k')

mask = (ace1hr['sai'] <= sai_thresh_low)
datachunk = ace1hr.loc[mask] 
column = 1

xdata = np.abs(datachunk['frac_of_yr'])
ydata = datachunk['Pinput'] #+ datachunk['P_B']
binned_median_plot(xdata, ydata, nxbins = nbins,fighandle = fig, axhandle = axs[row,column],
                  xmin = 0, xmax = 1, fmt='k')
ydata = datachunk['Pinput scaled'] #+ datachunk['P_B']
binned_median_plot(xdata, ydata, nxbins = nbins,fighandle = fig, axhandle = axs[row,column],
                  xmin = 0, xmax = 1, fmt='r')
ydata = datachunk['Pinput nocme'] #+ datachunk['P_B']
binned_median_plot(xdata, ydata, nxbins = nbins,fighandle = fig, axhandle = axs[row,column],
                  xmin = 0, xmax = 1, fmt='b')
axs[row,column].set_ylim((powplotmin,powplotmax))
axs[row,column].set_xlabel(r'Fraction of year')
axs[row,column].text(0.03,0.9,'(e)', fontsize = 14, transform=axs[row,column].transAxes, backgroundcolor = 'w')
axs[row,column].get_yaxis().set_ticklabels([])
yy = axs[row,column].get_ylim(); axs[row,column].plot([0.5, 0.5] ,yy ,'k')


mask = (ace1hr['sai'] >= sai_thresh_high)
datachunk = ace1hr.loc[mask] 
column = 2

xdata = np.abs(datachunk['frac_of_yr'])
ydata = datachunk['Pinput'] #+ datachunk['P_B']
binned_median_plot(xdata, ydata, nxbins = nbins,fighandle = fig, axhandle = axs[row,column],
                  xmin = 0, xmax = 1, fmt='k')
ydata = datachunk['Pinput scaled'] #+ datachunk['P_B']
binned_median_plot(xdata, ydata, nxbins = nbins,fighandle = fig, axhandle = axs[row,column],
                  xmin = 0, xmax = 1, fmt='r')
ydata = datachunk['Pinput nocme'] #+ datachunk['P_B']
binned_median_plot(xdata, ydata, nxbins = nbins,fighandle = fig, axhandle = axs[row,column],
                  xmin = 0, xmax = 1, fmt='b')
axs[row,column].set_ylim((powplotmin,powplotmax))
axs[row,column].set_xlabel(r'Fraction of year')
axs[row,column].text(0.03,0.9,'(f)', fontsize = 14, transform=axs[row,column].transAxes, backgroundcolor = 'w')
axs[row,column].get_yaxis().set_ticklabels([])
yy = axs[row,column].get_ylim(); axs[row,column].plot([0.5, 0.5] ,yy ,'k')


fig.tight_layout()
fig.savefig( os.path.join(fig_dir, 'pdyn_fracyr_medians.pdf'))  

# <codecell> Histograms of high and low latitude solar wind speeds
highlat = 6.5
lowlat = 1

Vthresh = 600

nhistbins = 50

dv = (vmax-vmin)/nhistbins
vbin_edges = np.arange(vmin - dv*5, vmax + dv*10, dv)
vbin_centres = (vbin_edges[1:]+vbin_edges[:-1])/2

fig, axs = plt.subplots(nrows =1, ncols = 3, figsize=(10, 4))




mask = (abs(ace1hr['Earth_lat'])  >= highlat) 
datachunk = ace1hr.loc[mask] 
counts, bins = np.histogram(datachunk['Vr'].dropna(), bins = vbin_edges)
N = np.sum(counts)
axs[0].plot(vbin_centres, counts/N)
print('All data')

pos = datachunk['Vr'] > Vthresh
print('High lat. V > ' + str(Vthresh) +'. p = ' +str(np.sum(pos)/N))

mask = (abs(ace1hr['Earth_lat'])  <= lowlat) 
datachunk = ace1hr.loc[mask] 
counts, bins = np.histogram(datachunk['Vr'].dropna(), bins = vbin_edges)
N = np.sum(counts)
axs[0].plot(vbin_centres, counts/N)

pos = datachunk['Vr'] > Vthresh
print('Low lat. V > ' + str(Vthresh) +'. p = ' +str(np.sum(pos)/N))




mask = (abs(ace1hr['Earth_lat'])  >= highlat) & (ace1hr['sai'] <= sai_thresh_low)
datachunk = ace1hr.loc[mask] 
counts, bins = np.histogram(datachunk['Vr'].dropna(), bins = vbin_edges)
N = np.sum(counts)
axs[1].plot(vbin_centres, counts/N)
print('Solar min')

pos = datachunk['Vr'] > Vthresh
print('High lat. V > ' + str(Vthresh) +'. p = ' +str(np.sum(pos)/N))

mask = (abs(ace1hr['Earth_lat'])  <= lowlat) & (ace1hr['sai'] <= sai_thresh_low)
datachunk = ace1hr.loc[mask] 
counts, bins = np.histogram(datachunk['Vr'].dropna(), bins = vbin_edges)
N = np.sum(counts)
axs[1].plot(vbin_centres, counts/N)

pos = datachunk['Vr'] > Vthresh
print('Low lat. V > ' + str(Vthresh) +'. p = ' +str(np.sum(pos)/N))




mask = (abs(ace1hr['Earth_lat'])  >= highlat) & (ace1hr['sai'] > sai_thresh_high)
datachunk = ace1hr.loc[mask] 
counts, bins = np.histogram(datachunk['Vr'].dropna(), bins = vbin_edges)
N = np.sum(counts)
axs[2].plot(vbin_centres, counts/N)
print('Solar max')

pos = datachunk['Vr'] > Vthresh
print('High lat. V > ' + str(Vthresh) +'. p = ' +str(np.sum(pos)/N))

mask = (abs(ace1hr['Earth_lat'])  <= lowlat) & (ace1hr['sai'] > sai_thresh_high)
datachunk = ace1hr.loc[mask] 
counts, bins = np.histogram(datachunk['Vr'].dropna(), bins = vbin_edges)
N = np.sum(counts)
axs[2].plot(vbin_centres, counts/N)

pos = datachunk['Vr'] > Vthresh
print('Low lat. V > ' + str(Vthresh) +'. p = ' +str(np.sum(pos)/N))

