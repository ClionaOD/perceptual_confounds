"""
Created on Thu Jan 14 14:01:06 2021

@author: Cliona O'Doherty, Hannah Craddock, Rhodri Cusack
"""

import pickle
import pandas as pd
from pandas.core.common import SettingWithCopyWarning
import numpy as np
import random
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix
import matplotlib.pyplot as plt
import warnings 

# HC function
def get_rest_df(rest_length):
    
    df_rest = pd.DataFrame()
    df_rest['onset'] = 22.525
    df_rest['duration'] = rest_length
    df_rest['trial_type'] = 'rest'
    df_rest['magnitude'] = 1.0
    
    return df_rest

# HC function
def get_df_events(all_events, rest_length = 0.0, sample_with_replacement=False, movie_length = None, delay = None, n_scans=None):
    '''Get concatenated events dataframe. 
    A period of rest in between each video is included: rest_length in seconds
    
    The function will randomly shuffle the order of movies each time unless sample_with_replacement
        is set to True in which case movies will be randomly sampled with replacement (for bootstrapping).
    '''

    if not n_scans:
        n_scans = (movie_length * len(all_events)) + ((delay + rest_length)*len(all_events))+10
        # Number of scans can't be odd
        if n_scans%2==1:
            n_scans+=1

    #List of videos + randomise
    list_videos = list(all_events.keys())
    if sample_with_replacement:
#        print('With replacement')
        list_videos = random.choices(list_videos,k=len(list_videos))
    else:
#        print('Not Shuffle')
        random.shuffle(list_videos) #Randomise movie order

    df_all_videos = pd.DataFrame()
    
    for idx, vid in enumerate(list_videos):      
        df_videoX = all_events[vid].copy()
        df_videoX.is_copy = None # get rid of setting with copy warning

        df_videoX['movie_source']=vid # it is important to remember where you came from
        
        # Find videos where the tags overrun
        overrun=(df_videoX['onset']+df_videoX['duration'])>22.9 # enforce 100 ms gap
        if overrun.any()>0:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
                df_videoX['duration'][overrun]=22.9-df_videoX['onset'][overrun]

        # Round off onset and duration
        df_videoX['onset'] = 0.01 + (df_videoX['onset']*51.0).round()/51.0 + idx*(movie_length + delay)
        df_videoX['duration'] = (df_videoX['duration']*51.0).round()/51.0 

        #Concatenate
        df_all_videos = df_all_videos.append(df_videoX, ignore_index = True)


    return df_all_videos, n_scans, list_videos


def get_design_matrix(events_dict=None, rest=0.00, hrf='spm', stacked_events = None, sample_with_replacement=False, tr=1.0, n_scans=None, movie_length = 22.524, delay = 23.0 - 22.524):
    #make design matrix for stacked events
    #param events_dict: the dict of movie event files (keys mov_name, values dataframe)


    #each time stack_events is called, the order of movies is randomised
    if stacked_events is None:
        stacked_events, n_scans, list_videos = get_df_events(events_dict, rest_length=rest, sample_with_replacement=sample_with_replacement, n_scans=n_scans, movie_length=movie_length, delay=delay)

    stacked_events = stacked_events.sort_values('onset', ignore_index= True)

    frame_times = np.arange(n_scans) * tr

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        X = make_first_level_design_matrix(frame_times, stacked_events, hrf_model=hrf)
    
    # for idx, vid in enumerate(list_videos):
    #     earliest=stacked_events[stacked_events['movie_source'] == vid]['onset'].min()
    #     latest=(stacked_events[stacked_events['movie_source'] == vid]['onset'] + stacked_events[stacked_events['movie_source'] == vid]['duration']).max()
    #     print(f'video {idx} earliest {earliest} latest {latest}')
    
 #   print(stacked_events)
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