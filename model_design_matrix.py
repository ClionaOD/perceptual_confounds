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

with open('./events_per_movie.pickle','rb') as f:
    all_events = pickle.load(f)

# make design matrices on a per movie basis
tr = 1.0
n_scans = 22
frame_times = np.arange(n_scans) * tr

indv_design_matrices = {k:None for k in all_events.keys()}
for vid, events in all_events.items():
    X = make_first_level_design_matrix(frame_times, events, hrf_model='spm')
    indv_design_matrices[vid] = X
    
    """
    fig, ax = plt.subplots()
    plot_design_matrix(X, ax=ax)
    ax.set_title(f'{vid[:-4]} design matrix', fontsize=12)
    plt.show()
    plt.close()
    """

#with open('./design_matrices_per_movie.pickle','wb') as f:
#    pickle.dump(indv_design_matrices, f)

#make design matrix for stacked events
rest = 0.00
tr = 1.0
n_scans = (22 * len(all_events)) + (rest*len(all_events))
frame_times = np.arange(n_scans) * tr

mov_list = list(all_events.keys())

all_vid_events = stack_events(all_events, mov_list, rest=rest)
X = make_first_level_design_matrix(frame_times, all_vid_events, hrf_model='spm')

#Efficiency
def efficiency(X):
    '''Calculate efficiency for a given design matrix (i.e a given video) '''       
    invXtX = np.linalg.inv(X.T.dot(X))
    #Constrasts - need to verify
    #Main(common) effect (1,1) vs differential effect (1, -1)
    contrasts = np.ones((invXtX.shape[0], 1)) #Need to verify: np.ones((invXtX.shape))
    efficiency = 1.0/(np.trace(contrasts.T.dot(invXtX).dot(contrasts)))
    
    return efficiency


    