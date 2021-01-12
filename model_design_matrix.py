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

