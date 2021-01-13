import pickle
import pandas as pd
import numpy as np
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix
import matplotlib.pyplot as plt

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
    plt.close()

with open('./design_matrices_per_movie.pickle','wb') as f:
    pickle.dump(all_design_matrices, f)

#Efficiency
def efficiency(X):
    '''Calculate efficiency for a given design matrix (i.e a given video) '''       
    invXtX = np.linalg.inv(X.T.dot(X))
    #Constrasts - need to verify
    #Main(common) effect (1,1) vs differential effect (1, -1)
    contrasts = np.ones((invXtX.shape[0], 1)) #Need to verify: np.ones((invXtX.shape))
    efficiency = 1.0/(np.trace(contrasts.T.dot(invXtX).dot(contrasts)))
    
    return efficiency


    