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
def get_df_events(all_events, rest_length = 2.0, sample_with_replacement=False):
    '''Get concatenated events dataframe. 
    A period of rest in between each video is included: rest_length in seconds
    
    The function will randomly shuffle the order of movies each time unless sample_with_replacement
        is set to True in which case movies will be randomly sampled with replacement (for bootstrapping).
    '''
    
    #List of videos + randomise
    list_videos = list(all_events.keys())
    if sample_with_replacement:
        print('With replacement')
        list_videos = random.choices(list_videos,k=len(list_videos))
    else:
        print('Not Shuffle')
        random.shuffle(list_videos) #Randomise movie order

    print(list_videos)

    #Params
    movie_length = 22.524
    delay = 23.0 - movie_length
    df_all_videos = pd.DataFrame()
    
    for idx, vid in enumerate(list_videos):      
        df_videoX = all_events[vid].copy()
        
        df_videoX['movie_source']=vid # it is important to remember where you came from
        
        # Find videos where the tags overrun
        overrun=(df_videoX['onset']+df_videoX['duration'])>22.9 # enforce 100 ms gap
        if overrun.any()>0:
            print(f'Video {idx} overruns {overrun.sum()}')
            df_videoX['duration'][overrun]=22.9-df_videoX['onset'][overrun]

        #Adjust onsets of event
        df_videoX['onset'] = 0.01 + (df_videoX['onset']*51.0).round()/51.0 + idx*(movie_length + delay)
#        df_videoX['onset'] = df_videoX['onset'] + idx*(movie_length + delay)

        df_videoX['duration'] = (df_videoX['duration']*51.0).round()/51.0 
        #Concatenate
        df_all_videos = df_all_videos.append(df_videoX, ignore_index = True)
    
    return df_all_videos, list_videos


def get_design_matrix(events_dict, rest=0.00, hrf='spm', sample_with_replacement=False, tr=1.0, n_scans=None):
    #make design matrix for stacked events
    #param events_dict: the dict of movie event files (keys mov_name, values dataframe)
    if not n_scans:
        n_scans = (23.0 * len(events_dict)) + (rest*len(events_dict))+10
    frame_times = np.arange(n_scans) * tr

    #each time stack_events is called, the order of movies is randomised
    if sample_with_replacement:
        stacked_events, list_videos = get_df_events(events_dict, rest_length=rest, sample_with_replacement=True)
    else:
        stacked_events, list_videos = get_df_events(events_dict, rest_length=rest)

    stacked_events = stacked_events.sort_values('onset', ignore_index= True)
    X = make_first_level_design_matrix(frame_times, stacked_events, hrf_model=hrf)
    
    # for idx, vid in enumerate(list_videos):
    #     earliest=stacked_events[stacked_events['movie_source'] == vid]['onset'].min()
    #     latest=(stacked_events[stacked_events['movie_source'] == vid]['onset'] + stacked_events[stacked_events['movie_source'] == vid]['duration']).max()
    #     print(f'video {idx} earliest {earliest} latest {latest}')
    
    print(stacked_events)
    return X
# walle
# 660.0    2.000000e+00
# 661.0    2.000000e+00
# 662.0    2.000000e+00
# 663.0    2.000000e+00
# 664.0    2.000000e+00
# 665.0    2.000000e+00
# 666.0    2.000000e+00
# 667.0    1.000000e+00
# 668.0   -7.305460e-15
# 669.0   -1.197587e-14
# 670.0    1.000000e+00
# 671.0    2.000000e+00
# 672.0    2.000000e+00
# 673.0    2.000000e+00
# 674.0    2.000000e+00
# 675.0    2.000000e+00
# 676.0    2.000000e+00
# 677.0    2.000000e+00
# 678.0    2.000000e+00
# 679.0   -1.094097e-14
# 680.0    1.000000e+00
# 681.0    2.000000e+00
# 682.0    2.000000e+00
# 683.0    2.000000e+00
# 684.0    2.000000e+00
# 685.0    2.000000e+00
# 686.0   -1.189133e-14
# 687.0   -1.201991e-14
# 688.0   -7.827517e-15
# 689.0    2.000000e+00
# 690.0   -1.216164e-15

#          onset  duration trial_type  magnitude movie_source
# 16885  299.160     0.620    animate        0.5    walle.mp4
# 16886  302.880     7.899    animate        0.5    walle.mp4
# 16887  311.072     6.688    animate        0.5    walle.mp4
# 16888  320.580     0.748    animate        0.5    walle.mp4
# 16930  299.000     0.606    animate        0.5    walle.mp4
# 16931  301.513     9.097    animate        0.5    walle.mp4
# 16932  312.940     4.625    animate        0.5    walle.mp4
# 16933  320.430     1.094    animate        0.5    walle.mp4
# 299.0    1.000000e+00
# 300.0   -4.632657e-15
# 301.0   -1.009989e-14
# 302.0    1.000000e+00
# 303.0    2.000000e+00
# 304.0    2.000000e+00
# 305.0    2.000000e+00
# 306.0    2.000000e+00
# 307.0    2.000000e+00
# 308.0    2.000000e+00
# 309.0    2.000000e+00
# 310.0    2.000000e+00
# 311.0   -8.647809e-15
# 312.0    1.000000e+00
# 313.0    2.000000e+00
# 314.0    2.000000e+00
# 315.0    2.000000e+00
# 316.0    2.000000e+00
# 317.0    2.000000e+00
# 318.0   -1.038616e-14
# 319.0   -1.031756e-14
# 320.0   -5.027524e-15
# 321.0    2.000000e+00
# 322.0    1.000000e+00

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