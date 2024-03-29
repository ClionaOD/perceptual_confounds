import pickle
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

from model_design_matrix import get_design_matrix
from nilearn.plotting import plot_design_matrix
import matplotlib.pyplot as plt

events = pd.read_pickle('../2.event_structuring/events_per_movie.pickle')

repeats = 5

desmats_conv = {i:{'design_matrix':None, 'corr_matrix':None} for i in range(repeats)}
desmats_nonconv = {i:{'design_matrix':None, 'corr_matrix':None} for i in range(repeats)}

for i in range(repeats):
    # create design matrix, randomly order movies
    hrf_desmat = get_design_matrix(events)
    desmats_conv[i]['design_matrix'] = hrf_desmat

    # get correlation matrix
    hrf_corr_mat = hrf_desmat.iloc[:,:-14].corr()
    desmats_conv[i]['corr_matrix'] = hrf_corr_mat

    # repeat without hrf convolution
    no_hrf_desmat = get_design_matrix(events, hrf=None)
    desmats_nonconv[i]['design_matrix'] = no_hrf_desmat

    no_hrf_corr_mat = no_hrf_desmat.iloc[:,:-14].corr()
    desmats_nonconv[i]['corr_matrix'] = no_hrf_corr_mat

    #plotting

    fig, ax = plt.subplots(figsize=(11.69,8.27))
    plot_design_matrix(hrf_desmat, ax=ax)
    ax.set_title(f'convolved design matrix', fontsize=12)
    plt.savefig(f'./design_matrices/convolved_run_{i+1}.png')
    #plt.show()
    plt.close()

    fig, ax = plt.subplots(figsize=(11.69,8.27))
    sns.heatmap(hrf_corr_mat, ax=ax)
    ax.set_title(f'correlation matrix - convolved', fontsize=12)
    plt.savefig(f'./design_matrices/correlation_matrices/convolved_run_{i+1}.png')
    #plt.show()
    plt.close()

    fig, ax = plt.subplots(figsize=(11.69,8.27))
    plot_design_matrix(no_hrf_desmat, ax=ax)
    ax.set_title(f'design matrix - no covolution', fontsize=12)
    plt.savefig(f'./design_matrices/no_conv_run_{i+1}.png')
    #plt.show()
    plt.close()

    fig, ax = plt.subplots(figsize=(11.69,8.27))
    sns.heatmap(no_hrf_corr_mat, ax=ax)
    ax.set_title(f'correlation matrix - no convolution', fontsize=12)
    plt.savefig(f'./design_matrices/correlation_matrices/no_conv_run_{i+1}.png')
    #plt.show()
    plt.close()

with open(f'./design_matrices/conv_{repeats}_runs.pickle','wb') as f:
    pickle.dump(desmats_conv, f)
with open(f'./design_matrices/not_conv_{repeats}_runs.pickle','wb') as f:
    pickle.dump(desmats_nonconv, f)


