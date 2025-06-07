#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 13:25:42 2025

Original script by Matt Palmer (m.d.palmer@bristol.ac.uk); adjusted & expanded by Aimée Slangen (aimee.slangen@nioz.nl)
"""

import matplotlib.pyplot as plt
import numpy as np
import csv


"""
Useful functions
"""
# this function reads in AR6 tide gauge ensemble
def read_tidegauge_gmsl(filename='AR6_GMSL_TG_ensemble_FGD.csv', key1='Central Estimate', 
                  key2='Total Unc. (1-sigma)\n', skip=2):
    """
    This function reads in the Ar6 tide gauge ensemble. 
    """
    f = open(filename, "r")
    lines = f.readlines()
    header = lines[skip - 1]
    keys = header.split(",")
    f.close()
    ncols = len(lines[-1].split(","))  # Get number of columns from the last line..
    nrows = len(lines) - skip  # Get number of data lines in text file
    data = np.zeros([nrows, ncols])  # Convention of rows then columns for Numpy arrays (?)
    for jj in range(nrows):
        for ii in range(ncols):
            data[jj, ii] = float(lines[jj + skip].split(",")[ii])
    data_dict = dict.fromkeys(keys)
    for kk, key in enumerate(keys):
        data_dict[key] = data[:, kk]
        
    yrs_tg = data_dict['Year']
    series_tg = data_dict[key1]
    error_tg  = data_dict[key2] #1 sigma error
        
    return yrs_tg, series_tg, error_tg

def read_tidegauge_indiv(filename='AR6_GMSL_reconstructions_FGD.csv', key1='Year', 
                  key2='CW2011', key3='HA2015',key4='DA2017',key5='DA2019',key6='FR2020',
                  key7='CW2011unc', key8='HA2015unc',key9='DA2017unc',key10='DA2019unc' ,key11='FR2020unc', 
                  skip=2):
    """
    This function reads in the AR6 individual tide gauge time series. 
    """
    f = open(filename, "r")
    lines = f.readlines()
    header = lines[skip - 1]
    keys = header.split(",")
    f.close()
    ncols = len(lines[-1].split(","))  # Get number of columns from the last line..
    nrows = len(lines) - skip  # Get number of data lines in text file
    data = np.zeros([nrows, ncols])  # Convention of rows then columns for Numpy arrays (?)
    for jj in range(nrows):
        for ii in range(ncols):
            data[jj, ii] = float(lines[jj + skip].split(",")[ii])
    data_dict = dict.fromkeys(keys)
    for kk, key in enumerate(keys):
        data_dict[key] = data[:, kk]
        
    yrs_tg = data_dict['Year']
    CW2011 = data_dict[key2]
    HA2015 = data_dict[key3]
    DA2019 = data_dict[key5]
    FR2020 = data_dict[key6]
        
    return yrs_tg, CW2011, HA2015, DA2019, FR2020

def read_altimeter_gmsl(filename='AR6_GMSL_altimeter_FGD.csv', key1='Central Estimate', 
                  key2='Uncertainty (1-sigma)\n', skip=2):
    """
    This function reads in the AR6 altimeter ensemble. 
    """
    f = open(filename, "r")
    lines = f.readlines()
    header = lines[skip - 1]
    keys = header.split(",")
    f.close()
    ncols = len(lines[-1].split(","))  # Get number of columns from the last line..
    nrows = len(lines) - skip  # Get number of data lines in text file
    data = np.zeros([nrows, ncols])  # Convention of rows then columns for Numpy arrays (?)
    for jj in range(nrows):
        for ii in range(ncols):
            data[jj, ii] = float(lines[jj + skip].split(",")[ii])
    data_dict = dict.fromkeys(keys)
    for kk, key in enumerate(keys):
        data_dict[key] = data[:, kk]
        
    yrs_2018 = data_dict['Year']
    mean_2018 = data_dict[key1]
    error_2018 = data_dict[key2] # 1-sigma error
   
        
    return yrs_2018, mean_2018, error_2018

def read_altimeter_gmsl_new(filename='altimetry_ens_1993_2024.csv', key1='mean', 
                  key2='std\n', skip=1):
    """
    This function reads in the new altimeter ensemble. 
    """
    f = open(filename, "r")
    lines = f.readlines()
    header = lines[skip - 1]
    keys = header.split(",")
    f.close()
    ncols = len(lines[-1].split(","))  # Get number of columns from the last line..
    nrows = len(lines) - skip  # Get number of data lines in text file
    data = np.zeros([nrows, ncols])  # Convention of rows then columns for Numpy arrays (?)
    for jj in range(nrows):
        for ii in range(ncols):
            data[jj, ii] = float(lines[jj + skip].split(",")[ii])
    data_dict = dict.fromkeys(keys)
    for kk, key in enumerate(keys):
        data_dict[key] = data[:, kk]
        
    yrs_2024 = data_dict['year']
    mean_2024 = data_dict[key1]
    error_2024 = data_dict[key2] # 1-sigma error
   
        
    return yrs_2024, mean_2024, error_2024

def read_altimeter_gmsl_indiv(filename='altimetry_indiv_estimates.csv', key1='mean', 
                  key2='AVISO',key3='NASA',key4='NOAA\n', skip=1):
    """
    This function reads in the new altimeter individual values. 
    """   
    f = open(filename, "r")
    lines = f.readlines()
    header = lines[skip - 1]
    keys = header.split(",")
    f.close()
    ncols = len(lines[-1].split(","))  # Get number of columns from the last line..
    nrows = len(lines) - skip  # Get number of data lines in text file
    data = np.zeros([nrows, ncols])  # Convention of rows then columns for Numpy arrays (?)
    for jj in range(nrows):
        for ii in range(ncols):
            data[jj, ii] = float(lines[jj + skip].split(",")[ii])
    data_dict = dict.fromkeys(keys)
    for kk, key in enumerate(keys):
        data_dict[key] = data[:, kk]
        
    yrs_indiv_2024 = data_dict['year']
    AVISO_2024 = data_dict[key2]
    NASA_2024 = data_dict[key3]
    NOAA_2024 = data_dict[key4]
   
        
    return yrs_indiv_2024, AVISO_2024, NASA_2024, NOAA_2024

def combine_TG_altimeter(tg_yrs, tg_series, tg_errors, alt_yrs, alt_series, alt_errors, fyr=1901.5, lyr=2024.5):
    """
    This function combines the tide gauge and altimeter timeseries and computes the 
    uncertainty in delta-GMSL relative to the first year of the timeseries. 
    """
    # Combined tide gauge and altimeter timeseries: 
    gmsl_yrs = np.arange(fyr, lyr+1, 1.0)
    gmsl_series = np.empty(len(gmsl_yrs))
    gmsl_errors = np.empty(len(gmsl_yrs))

    tmi = np.where((tg_yrs >= fyr) & (tg_yrs <= 1992.5))
    ami = np.where((alt_yrs >= 1993.5) & (alt_yrs <= lyr))
    gi1 = np.where((gmsl_yrs >= fyr ) & (gmsl_yrs <= 1992.5))
    gi2 = np.where((gmsl_yrs >= 1993.5) & (gmsl_yrs <= lyr))

    match_93 = tg_series[93]-alt_series[0] #connecting tg and altimetry in 1993

    gmsl_series[gi1] = tg_series[tmi]
    gmsl_series[gi2] = alt_series[ami]+match_93
    gmsl_errors[gi1] = tg_errors[tmi]
    gmsl_errors[gi2] = alt_errors[ami]
    
    # The delta error computes the total error in GMSL change for all timeseries points 
    # relative to the error in the first year (default = 1901). This is more smoothly varying. 
    delta_errors = gmsl_errors.copy()
    for ii in np.arange(len(gmsl_errors)):
        delta_errors[ii] = np.sqrt(np.square(delta_errors[0]) + np.square(delta_errors[ii]))
                              
    return gmsl_yrs, gmsl_series, gmsl_errors, delta_errors

def extract_core_period(yrs, series, errors, fyr, lyr):
    """
    This function extracts the core period as defined by fyr and lyr, padding the timeseries
    with zeros as needed. The year 1992.5 corresponds to availability of IMBIE data. 
    """
    core_yrs = np.arange(fyr, lyr+1)
    core_series = np.zeros(len(core_yrs))
    core_errors = np.zeros(len(core_yrs))
    vals, ind1, ind2 = np.intersect1d(core_yrs, yrs, return_indices=True)
    core_series[ind1] = series[ind2]
    core_errors[ind1] = errors[ind2]
    return core_yrs, core_series, core_errors


"""
Read in the TG and altimetry data, combine the timeseries
"""
tg_yrs, tg_series, tg_errors = read_tidegauge_gmsl()
alt_yrs_2024, alt_series_2024, alt_errors_2024 = read_altimeter_gmsl_new()
alt_yrs_2018, alt_series_2018, alt_errors_2018 = read_altimeter_gmsl()

## use WCRP values up to 2018 as in AR6, and splice on the 2018-2024 from new altimetry ensemble
alt_series_2024_WCRP = np.zeros([32])
alt_series_2024_WCRP[0:27] = alt_series_2018
alt_series_2024_WCRP[27:32] = alt_series_2024[27:32]-alt_series_2024[26]+alt_series_2018[26]
alt_series_2024 = alt_series_2024_WCRP
##

## use WCRP uncertainties up to 2018 and extend timeseries by repeating the final year
alt_errors_2024_WCRP = np.zeros([32])
alt_errors_2024_WCRP[0:27] = alt_errors_2018
alt_errors_2024_WCRP[27:32] = alt_errors_2024_WCRP[26]
alt_errors_2024 = alt_errors_2024_WCRP
##

#compute combined time series with AR6 up to 2018 and new altimetry from 2019
gmsl_yrs, gmsl_series, gmsl_errors, delta_errors = combine_TG_altimeter(tg_yrs, tg_series, tg_errors, alt_yrs_2024, alt_series_2024, alt_errors_2024)


#save data as csv file
headers = ['year','mean', 'std']
final_series=np.zeros([124,3])
final_series[:,0]=gmsl_yrs
final_series[:,1]=gmsl_series
final_series[:,2]=delta_errors
with open('IGCC_2024_GMSL_ensemble.csv','w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(headers)        
    writer.writerows(final_series)

#read individual time series for plotting
yrs_2024, AVISO_2024, NASA_2024, NOAA_2024 = read_altimeter_gmsl_indiv()
yrs_tg, CW2011, HA2015, DA2019, FR2020 = read_tidegauge_indiv()
    
    
#%% plot Figure A - the combined timeseries
fig = plt.figure(dpi=600, figsize=(8,6))
ax=fig.add_subplot()
fs=8


#ensemble & errors
ax.plot(gmsl_yrs[0:119], gmsl_series[0:119],'k' ,linewidth=2.0, label='AR6 Merged tide gauge + altimetry ensemble (1901-2018 w.r.t. 1995-2014)')
ax.fill_between(gmsl_yrs[0:119],(gmsl_series[0:119]-delta_errors[0:119]),(gmsl_series[0:119]+delta_errors[0:119]),color='k',alpha=0.1) # 1 sigma errors

#tg indiv
ax.plot(yrs_tg, CW2011, color='grey', linestyle = '--', linewidth=1.0, label='Tide gauge reconstructions w.r.t. 1995-2014 (CW2011, HA2015, DA2019, FR2020)')
ax.plot(yrs_tg, HA2015, color='grey', linestyle = '--', linewidth=1.0)
ax.plot(yrs_tg, DA2019, color='grey', linestyle = '--', linewidth=1.0)
ax.plot(yrs_tg, FR2020, color='grey', linestyle = '--', linewidth=1.0)

ax.plot(gmsl_yrs, gmsl_series,'k' ,linewidth=2.0)#replot on top for clarity

#add altimetry ensemble
ax.plot(gmsl_yrs[118:124], gmsl_series[118:124],'r' ,linewidth=2.0, label='Altimetry ensemble mean (2019-2024)')
ax.fill_between(gmsl_yrs[118:132],(gmsl_series[118:124]-delta_errors[118:124]),(gmsl_series[118:124]+delta_errors[118:124]),color='r',alpha=0.1) # 1 sigma errors

##satellite indiv wrt 2018
ax.plot(yrs_2024, AVISO_2024-AVISO_2024[26]+gmsl_series[118], color="#df0000", linestyle='--',linewidth=1.0, label='Satellite altimetry (AVISO, NOAA, NASA)')
ax.plot(yrs_2024, NASA_2024-NASA_2024[26]+gmsl_series[118], color="#df0000", linestyle='--',linewidth=1.0)
ax.plot(yrs_2024, NOAA_2024-NOAA_2024[26]+gmsl_series[118], color="#df0000", linestyle='--',linewidth=1.0)

#making the plot pretty
ax.set_title('a) Global mean sea-level rise', fontsize=fs+2)
ax.legend(loc='upper left', fontsize=fs, frameon=False)
ax.set_ylabel('GMSLR (mm)', fontsize=fs+2)
ax.set_xlabel('Time (years)', fontsize=fs+2)
ax.set_xlim(1901.5, 2024.5)

fig.tight_layout()
fig.savefig('SLC_figA_ts_IGCC24.png', bbox_inches='tight')


#%% Plot Figure B - rates

# compute total GMSL change, rates, and errors, and plot trends as bars

#use these periods
fyr=[1901.5, 1971.5, 1993.5, 2006.5] 
lyr=[2006.5, 2018.5, 2024.5]

trends = np.zeros([12, 4])
count=0
dp =2

for jj in fyr:
    for ii in lyr:
        if jj < 1993.5:
            # Read in TG timeseries for 1901-1993, from the combined timeseries
            sbyrs, sbseries, sberror = extract_core_period(tg_yrs, tg_series, tg_errors, fyr=jj, lyr=1993.5)
            del1 = sbseries[-1] - sbseries[0]
            err1 = np.sqrt(sberror[-1]**2 + sberror[0]**2) #1 sigma errors
            
            # Read in altimeter assessment timeseries for 1993 onwards, from the combined timeseries
            sbyrs, sbseries, sberror = extract_core_period(alt_yrs_2024, alt_series_2024, alt_errors_2024, fyr=1993.5, lyr=ii)
            del2 = sbseries[-1] - sbseries[0]
            err2 = np.sqrt(sberror[-1]**2 + sberror[0]**2)
            
            delta = del1 + del2
            error = np.sqrt(err1**2 + err2**2) #1 sigma errors   
                    
            label = str(int(jj)) + '-' + str(int(ii)) + ':'
            print(label,np.round(delta, decimals = dp), np.round(delta-(error*1.645), decimals=dp), np.round(delta+(error*1.645), decimals=dp))
        
            rate = delta/(ii-jj)
            rate_err = error/(ii-jj)
            print(np.round(rate, decimals=dp), np.round(rate-(rate_err*1.645), decimals=dp), np.round(rate+(rate_err*1.645), decimals=dp))

            trends[count,:] = (jj, ii, rate, rate_err*1.645) #very likely range
            count = count+1

        elif jj >= 1993.5: 
            # Read in altimeter assessment timeseries for 1993 onwards, from the combined timeseries
            sbyrs, sbseries, sberror = extract_core_period(alt_yrs_2024, alt_series_2024, alt_errors_2024, fyr=jj, lyr=ii)
            del2 = sbseries[-1] - sbseries[0]
            err2 = np.sqrt(sberror[-1]**2 + sberror[0]**2)
        
            delta = del2
            error = err2
                    
            label = str(int(jj)) + '-' + str(int(ii)) + ':'
            print(label,np.round(delta, decimals = dp), np.round(delta-(error*1.645), decimals=dp), np.round(delta+(error*1.645), decimals=dp))
        
            rate = delta/(ii-jj)
            rate_err = error/(ii-jj)
            print(np.round(rate, decimals=dp), np.round(rate-(rate_err*1.645), decimals=dp), np.round(rate+(rate_err*1.645), decimals=dp))
 
            trends[count,:] = (jj, ii, rate, rate_err*1.645)#very likely range
            count = count+1
            
# compute 20-yr trends
fyr=[1975.5, 1985.5, 1995.5, 2005.5, 2015.5] 
lyr=[1994.5, 2004.5, 2014.5, 2024.5, 2024.5]

trends_20y = np.zeros([5, 4])
count=0

for jj in fyr:
    ii = float(lyr[count])
    sbyrs, sbseries, sberror = extract_core_period(gmsl_yrs, gmsl_series, gmsl_errors, fyr=jj, lyr=ii)
    delta = sbseries[-1] - sbseries[0]
    error = np.sqrt(sberror[-1]**2 + sberror[0]**2)
    rate = delta/(ii-jj)
    rate_err = error/(ii-jj)
    trends_20y[count,:] = (jj, ii, rate, rate_err*1.645) #very likely range
    count = count+1


# plotting error bar trend figure
fig2 = plt.figure(dpi=600, figsize=(8,6))
ax2=fig2.add_subplot()
fs=8

alpha = 0.3
ar6_syrs = [1971., 1971., 2006.]
ar6_eyrs = [2018., 2006., 2018.]
IGCC_syrs=[1975., 1985., 1995., 2005., 2015.] 
IGCC_eyrs=[1994., 2004., 2014., 2024., 2024.]

# Central estimates and +/- error for very likely ranges
ar6_dict = {'1971-2018':[trends[4,2], trends[4,3]],#[2.33,0.79], 
            '1971-2006':[trends[3,2], trends[3,3]],# 
            '2006-2018':[trends[9,2], trends[9,3]]}#[3.69,0.48]}

IGCC_dict = {'1971-2018':[trends[4,2],trends[4,3]], 
            '1971-2006':[trends[3,2],trends[3,3]], 
            '2006-2018':[trends[9,2],trends[9,3]], 
            '1975-1994':[trends_20y[0,2],trends_20y[0,3]], 
            '1985-2004':[trends_20y[1,2],trends_20y[1,3]], 
            '1995-2014':[trends_20y[2,2],trends_20y[2,3]], 
            '2005-2024':[trends_20y[3,2],trends_20y[3,3]],
            '2015-2024':[trends_20y[4,2],trends_20y[4,3]]}

labels=[] # Define an empty list for year labels

for ss1, syr in enumerate(ar6_syrs):
    xpts = [ss1-0.3, ss1+0.3]
    color = 'tab:red'
    eyr = ar6_eyrs[ss1]
    label = str(int(syr)) + '-' + str(int(eyr))
    labels.append(label)
    delta, error = ar6_dict[str(int(syr))+'-'+str(int(eyr))]
    label1 = None
    label2 = None
    if ss1 == 0:
        label1 = 'IPCC AR6'
        label2 = 'This study'
    ax2.fill_between(xpts, delta-error, delta+error, facecolor='blue', alpha=alpha, label=label1)
    ax2.plot(xpts, [delta, delta], linewidth=2.0, color='blue')   

    delta, error = IGCC_dict[str(int(syr))+'-'+str(int(eyr))]
    ax2.fill_between(xpts, delta-error, delta+error, facecolor='red', alpha=alpha, label=label2)
    ax2.plot(xpts, [delta, delta], linewidth=2.0, color='red')
ss1 +=1

for ss, syr in enumerate(IGCC_syrs):
    xpts = [ss+ss1-0.3, ss+ss1+0.3]
    color = 'tab:red'
    eyr = IGCC_eyrs[ss]
    label = str(int(syr)) + '-' + str(int(eyr))
    labels.append(label)
    delta, error = IGCC_dict[str(int(syr))+'-'+str(int(eyr))]
    ax2.fill_between(xpts, delta-error, delta+error, facecolor=color, alpha=alpha)
    ax2.plot(xpts, [delta, delta], linewidth=2.0, color=color)


#making the plot pretty
xmin, xmax = -1, 8
for val in [-2., -1., 0., 1., 2.,3., 4., 5. ]:
    ax2.plot([xmin, xmax],[val, val], color='silver', linestyle=':', linewidth=0.5)

ax2.set_title('b) Global mean sea-level rise rates', fontsize=fs+2)
#ax2.set_title('b) Global mean sea-level change rates', fontsize=fs+2)
ax2.legend(loc='upper left', fontsize=fs, frameon=False)
ax2.set_ylabel('GMSLR rate (mm/yr)', fontsize=fs+2)
ax2.set_ylim(-1, 5)
ax2.set_xlim(-0.5, 7.5)
ax2.set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
ax2.set_xticklabels(labels,rotation=45, fontsize=fs)


fig2.tight_layout()
fig2.savefig('SLC_figB_trends_IGCC24.png', bbox_inches='tight')


