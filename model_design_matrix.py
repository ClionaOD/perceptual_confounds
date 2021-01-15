"""
Created on Thu Jan 14 14:01:06 2021

@author: Cliona O'Doherty, Hannah Craddock
"""

import pickle
import pandas as pd
import numpy as np
import random
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix
import matplotlib.pyplot as plt

# HC function
def get_rest_df(rest_length):
    
    df_rest = pd.DataFrame()
    df_rest['onset'] = 22.525
    df_rest['duration'] = rest_length
    df_rest['trial_type'] = 'rest'
    df_rest['magnitude'] = 1.0
    
    return df_rest

# HC function
def get_df_events(all_events, rest_length = 2.0):
    '''Get concatenated events dataframe. A period of rest in between each video is included'''
    
    #List of videos + randomise
    list_videos = list(all_events.keys())
    random.shuffle(list_videos) #Randomise movie order
    
    #Params
    movie_length = 22.524
    delay = 0.001
    df_rest = get_rest_df(rest_length)
    df_events_concat = pd.DataFrame()
    
    for idx, vid in enumerate(list_videos):      
        df_eventX = all_events[vid]
        df_temp = pd.concat([df_eventX, df_rest])
        #Adjust onsets of event
        df_temp['onset'] = df_temp['onset'] + idx*(movie_length + rest_length + delay)
        
        #Concatenate
        df_events_concat = pd.concat([df_events_concat,  df_temp])
    
    return df_events_concat


def get_design_matrix(events_dict, rest=0.00, hrf='spm'):
    #make design matrix for stacked events
    #param events_dict: the dict of movie event files (keys mov_name, values dataframe)
    tr = 1.0
    n_scans = (22 * len(events_dict)) + (rest*len(events_dict))
    frame_times = np.arange(n_scans) * tr

    #each time stack_events is called, the order of movies is randomised
    stacked_events = get_df_events(events_dict, rest_length=rest)
    X = make_first_level_design_matrix(frame_times, stacked_events, hrf_model=hrf)
    
    return X

#Efficiency
def efficiency_calc(X, contrast_vec):
    '''Calculate efficiency for a given design matrix (i.e a given video) '''       
    invXtX = np.linalg.inv(X.T.dot(X))
    efficiency = np.trace((1.0/ contrast_vec.T.dot(invXtX).dot(contrast_vec)))
    
    return efficiency

def get_contrasts(desmat):
    conditions = desmat.columns.tolist()
    
    contrast_vec = np.zeros((len(conditions),1))

    animate_idx = conditions.index('animate')
    inanimate_big_idx = conditions.index('inanimate_big')
    inanimate_small_idx = conditions.index('inanimate_small')

    contrast_vec[animate_idx] = 1
    contrast_vec[inanimate_big_idx] = -1
    contrast_vec[inanimate_small_idx] = -1

    return contrast_vec

if __name__ == "__main__":

    with open('./events_per_movie.pickle','rb') as f:
        all_events = pickle.load(f)

    while len(all_events) > 8:
        all_vids_desmat = get_design_matrix(all_events)
        all_contrast = get_contrasts(all_vids_desmat)
        all_vids_efficiencies = efficiency_calc(all_vids_desmat, all_contrast)
        print(f'with {len(all_events)} videos, efficiency = {all_vids_efficiencies}')

        loa_efficiencies = {k:None for k in all_events.keys()}
        for mov in all_events.keys():
            leave_one_out_events = {k:v for k,v in all_events.items() if not k==mov}
            loa_desmat = get_design_matrix(leave_one_out_events)
            loa_contrasts = get_contrasts(loa_desmat)
            loa_efficiencies[mov] = efficiency_calc(loa_desmat,loa_contrasts)

        efficiency_df = pd.DataFrame.from_dict({k:[v] for k,v in loa_efficiencies.items()})
        #efficiency_df.loc[1] = all_vids_efficiencies - efficiency_df.loc[0,:].values
        drop_mov = efficiency_df.idxmax(axis=1).loc[0]

        print(f'removing {drop_mov} lowers efficiency by the least to {efficiency_df[drop_mov][0]}')

        all_events.pop(drop_mov)