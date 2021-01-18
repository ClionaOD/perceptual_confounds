import pickle
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import scipy.cluster.hierarchy as sch

def hierarchical_clustering(in_matrix, label_list, outpath=None, ax=None):
    matrix = copy.copy(in_matrix)
    np.fill_diagonal(matrix, 0)
    if ax is not None:
        dend = sch.dendrogram(sch.linkage(matrix, method='ward'), 
            ax=ax, 
            labels=label_list, 
            orientation='right'
        )
    else:   
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
    if ax is None:
        plt.close()

    cluster_order = dend['ivl']

    return cluster_order

with open('./design_matrices/conv_5_runs.pickle',"rb") as f:
    conv = pickle.load(f)

with open('./design_matrices/not_conv_5_runs.pickle',"rb") as f:
    not_conv = pickle.load(f)

for run, results in conv.items():
    fig, (ax1, ax2) = plt.subplots(figsize=(13,8.27), ncols=2)
    
    order = hierarchical_clustering(results['corr_matrix'].values,
        results['corr_matrix'].columns, 
        ax=ax1
        )
    
    order.reverse()
    new_corr_mat = results['corr_matrix'].reindex(index=order,columns=order)

    
    sns.heatmap(new_corr_mat, ax=ax2)
    ax2.set_title(f'clustered correlation matrix - convolved', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'./clustered_corr_matrices/convolved_run_{run+1}.pdf')
    #plt.show()
    plt.close()

for run, results in not_conv.items():
    fig, (ax1, ax2) = plt.subplots(figsize=(13,8.27), ncols=2)
    
    order = hierarchical_clustering(results['corr_matrix'].values,
        results['corr_matrix'].columns, 
        ax=ax1
        )
    
    order.reverse()
    new_corr_mat = results['corr_matrix'].reindex(index=order,columns=order)

    
    sns.heatmap(new_corr_mat, ax=ax2)
    ax2.set_title(f'clustered correlation matrix - not convolved', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'./clustered_corr_matrices/not_convolved_run_{run+1}.pdf')
    #plt.show()
    plt.close()