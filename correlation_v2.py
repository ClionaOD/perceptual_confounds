# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 10:35:32 2021

@author: Hannah Craddock
"""

import seaborn as sns
import pickle
import pandas as pd
import numpy as np
import random
from nilearn.glm.first_level import make_first_level_design_matrix
#from nilearn.plotting import plot_design_matrix
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)

#Repeat time
ver = 1

#Plot properties
%matplotlib qt

MEDIUM_SIZE = 15
BIGGER_SIZE = 19

plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#Data
all_events = pd.read_pickle('./events_per_movie.pickle')

#Correlation

def get_df_all_videos(all_events):
    '''Get concatenated events dataframe. Order of movies is randomised each time'''
    
    #List of videos + randomise
    list_videos = list(all_events.keys())
    random.shuffle(list_videos) #Randomise movie order
    
    #Params
    movie_length = 22.524
    delay = 0.001
    df_all_videos = pd.DataFrame()
    
    for idx, vid in enumerate(list_videos):      
        df_videoX = all_events[vid].copy()
        #Adjust onsets of event
        df_videoX['onset'] = df_videoX['onset'] + idx*(movie_length + delay)
     
        #Concatenate
        df_all_videos = pd.concat([df_all_videos,  df_videoX])
    
    return df_all_videos

def get_design_matrix(all_events, rest=0.00, hrf='spm'):
    '''Make design matrix of concatenated videos'''
    
    #Concatenated videos - randomised order each time
    df_all_videos = get_df_all_videos(all_events)
    
    #Get 
    tr = 1.0
    n_scans = (22 * len(all_events)) + (rest*len(all_events))
    frame_times = np.arange(n_scans) * tr

    X = make_first_level_design_matrix(frame_times, df_all_videos, hrf_model=hrf)
    
    return X

def correlation_matrix(X, convolved, n, path_save):
    '''Get correlation matrix of design matrix of the movies '''
    
    #Processing of design matrix
    X.drop(['constant'], axis = 1, inplace = True)  #Remove additional columns - not in videos for correlation
    X = X[X.columns.drop(list(X.filter(regex='drift')))]   
    #Correlation matrix
    corrMatrix = X.corr()
    
    #Figure
    fig, ax = plt.subplots(figsize=(11.5, 9))
    sns.heatmap(corrMatrix, ax=ax)
    ax.set_title('correlation matrix - {}'.format(convolved), fontsize=12)
    plt.savefig(path_save + '\{}_{}.png'.format(convolved, n+1), dpi=300, bbox_inches = "tight")
    #plt.savefig('.\design_matrices\correlation_matrices_hc\{}_{}.png'.format(convolved, n+1))
    plt.show()
    
    return corrMatrix
      
    
def repeat_corr(all_events, n_repeats, path_save):
    '''Get correlation matrix of movies, iterate n_repeat times with different randomised order of movies '''
    
    #Storage of results
    results_conv = {n:{'design_matrix':None, 'corr_matrix':None} for n in range(n_repeats)}
    results_nonconv = {n:{'design_matrix':None, 'corr_matrix':None} for n in range(n_repeats)}
    
    for n in range(n_repeats):
        
        #1. Convolved WITH HRF 
        convolved = 'convolved_hrf'
        X = get_design_matrix(all_events) #Get design matrix of concatenated events. hrf = 'spm'
        corr = correlation_matrix(X, convolved, n, path_save) 
        #Save results
        results_conv[n]['design_matrix'] = X
        results_conv[n]['corr_matrix'] = corr
        
    
        #2. Not covovled with hrf
        convolved = 'not_convolved'
        hrf = None
        X = get_design_matrix(all_events, 0, hrf)
        corr = correlation_matrix(X, convolved, n, path_save) 
        #Save results
        results_nonconv[n]['design_matrix'] = X
        results_nonconv[n]['corr_matrix'] = corr
    
    #Save final results
    with open(path_save + '/conv_results_{}.pickle'.format(ver),'wb') as f:
        pickle.dump(results_conv, f)
    with open(path_save + '/nonconv_results_{}.pickle'.format(ver),'wb') as f:
        pickle.dump(results_nonconv, f)
        

#Repeat correlation
path_save = './design_matrices/correlation_matrices_hc/'
n_repeats = 5
repeat_corr(all_events, n_repeats, path_save)


#Get df all events
df_all_videos = get_df_all_videos(all_events)      
#Corr matrix
X = get_correlation_matrix(df_all_videos) 
corrMatrix =  get_correlation_matrix(df_all_videos)   