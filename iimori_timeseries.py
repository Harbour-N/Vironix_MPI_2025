
'''
TO USE:
1. make iimori folder in project folder if it doesn't exist.
2. dump the data (xlsx file) there 
    See: https://datadryad.org/dataset/doi:10.5061/dryad.kq23s
    associated paper: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0190493

3. On first run, this script will try to save a parquet format of the same 
file to the same folder; which should accelerate file load on subsequent imports.

'''

import pandas as pd
import numpy as np
from scipy import stats

import os

################

def sim_egfr_drop_event(vec, p=0.1, multiplier=stats.uniform(0.4,0.6)):
    '''Given a vector of time series of eGFR values, 
    probabilistically sample an event which permanently drops the remaining 
    values by a fixed amount.
    
    Input: vec: numpy array
            p: float, probability of event at a time point (default: 0.1)
            
            multiplier: callable function from scipy.stats 
                for random variable for a multiplicative 
                amount the eGFR drops by. (default: uniform random on (0.4, 1) )
                (NOTE: stats.uniform is parameterized by (min, min+length).)
                (NOTE: more generically, can have any function that is callable 
                with a multiplier.rvs() method that outputs a numerical value.)
    '''
    vec_new = np.array(vec)
    for i in range(len(vec)):
        if np.random.uniform()<p:
            vec_new[i:] = vec_new[i:]*multiplier.rvs()
            return vec_new
    return vec_new # if no event, then return original array

def sim_aki_event(vec, p=0.1, multiplier=stats.uniform(0.2,0.3)):
    '''Given a vector of time series of eGFR values, 
    probabilistically sample an event which drops the eGFR at a single 
    time point. The rest of the values remain at their original values.
    This is meant to simulate an AKI (acute kidney injury) event.
    
    Input: vec: numpy array
            p: float, probability of event at a time point (default: 0.1)
            
            multiplier: callable function from scipy.stats 
                for random variable for a multiplicative 
                amount the eGFR drops by. (default: uniform random on (0.2, 0.5) )
                (NOTE: stats.uniform is parameterized by (min, min+length).)
                (NOTE: more generically, can have any function that is callable 
                with a multiplier.rvs() method that outputs a numerical value.)
    TODO: explore how severe the acute drops are in the data we've seen, or 
    described in literature.
    '''
    vec_new = np.array(vec)
    for i in range(len(vec)):
        if np.random.uniform()<p:
            vec_new[i] = vec[i]*multiplier.rvs()
            return vec_new
    return vec_new # if no event, then return original array
#


# See: https://datadryad.org/dataset/doi:10.5061/dryad.kq23s
# associated paper: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0190493
if os.path.exists('./iimori/ROUTE_proteinuria_dataset.pq'):
    df_orig = pd.read_parquet('./iimori/ROUTE_proteinuria_dataset.pq')
else:
    df_orig = pd.read_excel('./iimori/ROUTE_proteinuria_dataset.xlsx')
    df_orig.to_parquet('./iimori/ROUTE_proteinuria_dataset.pq')

# TODO: what is ID?

# eGFR timeseries.
cols = [
    'eGFR(0M)',
    'eGFR(6M)',
    'eGFR(12M)',
    'eGFR(18M)',
    'eGFR(24M)',
    'eGFR(30M)',
    'eGFR(36M)'
]

mask = df_orig[cols].notna().all(axis=1)

# filter data down to those with complete time series when focused on 
# eGFR over the course of three years.
#
# TODO: address bias in throwing out patients with missingness.
df = (df_orig[cols])[mask]

if __name__=="__main__":
    from matplotlib import pyplot as plt
    from matplotlib import ticker
    
    plt.style.use('fivethirtyeight')
    plt.rcParams.update({'font.size': 14})
    
    t = np.arange(0, 36+1, 6)

    fig,ax = plt.subplots(1,2, figsize=(10,5), sharex=True, sharey=True, constrained_layout=True)

    #for i in range(df_sub.shape[0]):
    row = df_sub.iloc[0].values

    row_aki = sim_aki_event(row, p=0.5)
    row_bigdrop = sim_egfr_drop_event(row, p=0.5)

    ax[0].plot(t, row, marker='.', lw=2, label='original')
    ax[0].plot(t, row_aki, marker='.', lw=2, label='simulated AKI')

    ax[1].plot(t, row, marker='.', lw=2, label='original')
    ax[1].plot(t, row_bigdrop, marker='.', lw=2, label='simulated perm. drop')

    for i in range(2):
        ax[i].legend(loc='upper right')
        ax[i].xaxis.set_major_locator(ticker.MultipleLocator(6))
        ax[i].set(xlabel='t (months)', ylabel='eGFR')

    fig.savefig('output/simulated_aki.png', bbox_inches='tight')
    fig.savefig('output/simulated_aki.pdf', bbox_inches='tight')
    fig.show()

