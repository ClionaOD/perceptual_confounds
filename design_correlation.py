import pickle
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

from model_design_matrix import get_design_matrix
from nilearn.plotting import plot_design_matrix
import matplotlib.pyplot as plt

# Note: for loop below causing strange error in design matrices.
# For now, run the script through with repeats=1 and specify the run number. 
run = 5

events = pd.read_pickle('./events_per_movie.pickle')

repeats = 1

desmats_conv = {i:{'design_matrix':None, 'corr_matrix':None} for i in range(repeats)}
desmats_nonconv = {i:{'design_matrix':None, 'corr_matrix':None} for i in range(repeats)}

for i in range(repeats):
    X = get_design_matrix(events)
    desmats_conv[i]['design_matrix'] = X

    hrf_corr_mat = X.iloc[:,:-14].corr()
    desmats_conv[i]['corr_matrix'] = hrf_corr_mat

    #in get_design_matrix, nilearn make_first_level_design_matrix has
    #  option to include hrf=None. Unsure if this is working correctly as results 
    #  are highly similar.
    X_no_hrf = get_design_matrix(events, hrf=None)
    desmats_nonconv[i]['design_matrix'] = X_no_hrf

    corr_mat = X_no_hrf.corr()
    desmats_nonconv[i]['corr_matrix'] = corr_mat

    fig, ax = plt.subplots(figsize=(11.69,8.27))
    plot_design_matrix(X, ax=ax)
    ax.set_title(f'convolved design matrix', fontsize=12)
    plt.savefig(f'./design_matrices/convolved_{run}.png')
    plt.show()
    plt.close()

    fig, ax = plt.subplots(figsize=(11.69,8.27))
    sns.heatmap(hrf_corr_mat, ax=ax)
    ax.set_title(f'correlation matrix - convolved', fontsize=12)
    plt.savefig(f'./design_matrices/correlation_matrices/convolved_{run}.png')
    plt.show()
    plt.close()

    fig, ax = plt.subplots(figsize=(11.69,8.27))
    plot_design_matrix(X, ax=ax)
    ax.set_title(f'design matrix - no covolution', fontsize=12)
    plt.savefig(f'./design_matrices/no_conv_{run}.png')
    plt.show()
    plt.close()

    fig, ax = plt.subplots(figsize=(11.69,8.27))
    sns.heatmap(hrf_corr_mat, ax=ax)
    ax.set_title(f'correlation matrix - no convolution', fontsize=12)
    plt.savefig(f'./design_matrices/correlation_matrices/no_conv_{run}.png')
    plt.show()
    plt.close()

with open(f'./design_matrices/conv_run_{run}.pickle','wb') as f:
    pickle.dump(desmats_conv, f)
with open(f'./design_matrices/not_conv_run_{run}.pickle','wb') as f:
    pickle.dump(desmats_nonconv, f)


