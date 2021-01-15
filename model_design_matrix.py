import pickle
import pandas as pd
import numpy as np
import random
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix
import matplotlib.pyplot as plt

def stack_events(events, mov_list, rest=0.00):
    """take movie events files and concatenate them in a random order, 
        this simulates an experiments' event file for a given run.
    params:
        events = dict of events files, keys=movie_titles,values=dataframe
        mov_list = list of movies to stack, each element should be a key in events
        rest = size of the gap to include between displayed movies
    """
    #shuffle the order of the movies
    random.shuffle(mov_list)
    
    df = pd.DataFrame(columns=events[mov_list[0]].columns)
    
    for idx, mov in enumerate(mov_list):
        mov_events = events[mov]
        
        offset = (22.524*idx) + (rest*idx)
        mov_events['onset'] = mov_events['onset'].values + offset
        df = pd.concat([df, mov_events])
        
        #get value in sec at which previous movie ended - all movs have longest duration=22.524
        end_mov = 22.524 * (idx+1)
        
        #add rest
        rest_ = {k:[] for k in mov_events.columns}
        rest_['onset'] = [end_mov+0.001]; rest_['duration'] = [rest] ; rest_['trial_type'] = ['rest'] ; rest_['magnitude'] = [1]
        rest_event = pd.DataFrame.from_dict(rest_)
        df = pd.concat([df, rest_event])
    
    return df

def get_design_matrix(events_dict, rest=0.00, hrf='spm'):
    #make design matrix for stacked events
    #param events_dict: the dict of movie event files (keys mov_name, values dataframe)
    tr = 1.0
    n_scans = (22 * len(events_dict)) + (rest*len(events_dict))
    frame_times = np.arange(n_scans) * tr

    mov_list = list(events_dict.keys())

    #each time stack_events is called, the order of movies is randomised
    stacked_events = stack_events(events_dict, mov_list, rest=rest)
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