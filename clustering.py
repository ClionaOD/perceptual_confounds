import pickle
import copy
import matplotlib.pyplot as plt
import seaborn as sns
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

with open('./design_matrices/conv_5_runs.pickle',"rb") as f:
    conv = pickle.load(f)

with open('./design_matrices/not_conv_5_runs.pickle',"rb") as f:
    not_conv = pickle.load(f)

for run, results in conv.items():
    order = hierarchical_clustering(results['corr_matrix'].values,
        results['corr_matrix'].columns, 
        outpath=f'./clustered_corr_matrices/dendro_convolved_run_{run+1}.png'
        )
    
    new_corr_mat = results['corr_matrix'].reindex(index=order,columns=order)

    fig, ax = plt.subplots(figsize=(11.69,8.27))
    sns.heatmap(new_corr_mat, ax=ax)
    ax.set_title(f'clustered correlation matrix - convolved', fontsize=12)
    plt.savefig(f'./clustered_corr_matrices/corr_convolved_run_{run+1}.png')
    #plt.show()
    plt.close()

for run, results in not_conv.items():
    order = hierarchical_clustering(results['corr_matrix'].values,
        results['corr_matrix'].columns, 
        outpath=f'./clustered_corr_matrices/dendro_not_convolved_run_{run+1}.png'
        )
    
    new_corr_mat = results['corr_matrix'].reindex(index=order,columns=order)

    fig, ax = plt.subplots(figsize=(11.69,8.27))
    sns.heatmap(new_corr_mat, ax=ax)
    ax.set_title(f'clustered correlation matrix - not convolved', fontsize=12)
    plt.savefig(f'./clustered_corr_matrices/corr_not_convolved_run_{run+1}.png')
    #plt.show()
    plt.close()