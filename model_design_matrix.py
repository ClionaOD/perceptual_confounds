import pickle
import pandas as pd
import numpy as np
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix
import matplotlib.pyplot as plt

#Plot properties
%matplotlib qt

SMALL_SIZE = 8
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#Data
with open('./events_per_movie.pickle','rb') as f:
    all_events = pickle.load(f)

tr = 1.0
n_scans = 22
frame_times = np.arange(n_scans) * tr



all_design_matrices = {k:None for k in all_events.keys()}
for vid, events in all_events.items():
    X = make_first_level_design_matrix(frame_times, events, hrf_model='glover')
    all_design_matrices[vid] = X
    
    fig, ax = plt.subplots()
    plot_design_matrix(X, ax=ax)
    ax.set_title(f'{vid[:-4]} design matrix', fontsize=12)
    plt.show()
    break 
    #plt.close()

with open('./design_matrices_per_movie.pickle','wb') as f:
    pickle.dump(all_design_matrices, f)


#*****************************************************
#Efficiency calculation

def efficiency_calc(X):
    '''Calculate efficiency for a given design matrix (i.e a given video) '''       
    invXtX = np.linalg.inv(X.T.dot(X))
    #Constrasts - need to verify
    #Main(common) effect (1,1) vs differential effect (1, -1)
    contrasts = np.ones((invXtX.shape)) #np.ones((invXtX.shape[0], 1)) #Need to verify: np.ones((invXtX.shape))
    efficiency = np.trace((1.0/ contrasts.T.dot(invXtX).dot(contrasts)))
    
    return efficiency

def get_efficiencies(dict_events):
    '''Return efficiency for each video in a dictionary'''
    dict_efficiencies = {k:None for k in dict_events.keys()}
    
    for vid, events in dict_events.items():
        X = make_first_level_design_matrix(frame_times, events, hrf_model='glover')
        dict_efficiencies[vid] =   efficiency_calc(X)
    
    return dict_efficiencies
 
#Apply
dict_efficiencies = get_efficiencies(all_events)

#Plot values
lists = sorted(dict_efficiencies.items()) # sorted by key, return a list of tuples
vid_names, efficiencies = zip(*lists) # unpack a list of pairs into two tuples
plt.barh(vid_names, efficiencies)
plt.title('Efficiency', fontsize=18)
plt.show()
plt.savefig('./Efficiency_plot')
    
    


    