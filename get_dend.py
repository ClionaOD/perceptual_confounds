import pickle
import copy
import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as sch

def hierarchical_clustering(in_matrix, label_list, outpath=None):
    matrix = copy.copy(in_matrix)
    np.fill_diagonal(matrix, 0)
    fig,ax = plt.subplots(figsize=(15,10))
    dend = sch.dendrogram(sch.linkage(matrix, method='ward'), 
        ax=ax, 
        labels=label_list, 
        orientation='right'
    )
    ax.tick_params(axis='x', labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if outpath:
        plt.savefig(outpath)
    plt.close()

    cluster_order = dend['ivl']

    return cluster_order