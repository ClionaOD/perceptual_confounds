# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 15:16:59 2021

@author: Hannah Craddock
"""

import pickle
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import os 

import scipy.cluster.hierarchy as sch

def hierarchical_clustering(in_matrix, label_list, method_clustering, outpath=None, ax=None):
    matrix = copy.copy(in_matrix)
    np.fill_diagonal(matrix, 0)
    if ax is not None:
        dend = sch.dendrogram(sch.linkage(matrix, method= method_clustering), 
            ax=ax, 
            labels=label_list, 
            orientation='right'
        )
    else:   
        fig,ax = plt.subplots(figsize=(15,10))
        dend = sch.dendrogram(sch.linkage(matrix, method= method_clustering), 
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

    cluster_order = dend['ivl'] #A list of labels corresponding to the leaf nodes.

    return cluster_order

def repeat_clustering_method(method_clustering, root_path, dir_save):
    
    #Data
    with open(root_path + 'conv_results_1.pickle',"rb") as f: # './design_matrices/conv_5_runs.pickle'
        conv = pickle.load(f) #Dictionary: key = run, values = design matrix and correlation matrix
    
    with open(root_path + 'nonconv_results_1.pickle',"rb") as f: # './design_matrices/not_conv_5_runs.pickle'
        not_conv = pickle.load(f)
    
    #Create directory
    dir_path = dir_save + method_clustering
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    for run, results in conv.items():
        print('run number = {}'.format(run))
        fig, (ax1, ax2) = plt.subplots(figsize=(11.69,8.27), ncols=2)
    
        order = hierarchical_clustering(results['corr_matrix'].values,
            results['corr_matrix'].columns, method_clustering,
            ax=ax1
            )
        
        order.reverse() #Why is the order revervsed?
        new_corr_mat = results['corr_matrix'].reindex(index=order,columns=order) #Corr matrix, rows = columns 
    
        
        sns.heatmap(new_corr_mat, ax=ax2)
        ax2.set_title(f'clustered correlation matrix - convolved')       
        plt.tight_layout()
        
        #Save figure
        plt.savefig(dir_path + '/convolved_run_{}.pdf'.format(run+1))
        #plt.show()
        plt.close()
    
    for run, results in not_conv.items():
        fig, (ax1, ax2) = plt.subplots(figsize=(11.69,8.27), ncols=2)
        
        order = hierarchical_clustering(results['corr_matrix'].values,
            results['corr_matrix'].columns, method_clustering,
            ax=ax1
            )
        
        order.reverse()
        new_corr_mat = results['corr_matrix'].reindex(index=order,columns=order)
    
        
        sns.heatmap(new_corr_mat, ax=ax2)
        ax2.set_title(f'clustered correlation matrix - not convolved')
        
        plt.tight_layout()
        plt.savefig(dir_path + '/not_convolved_run{}.pdf'.format(run + 1))
        #plt.show()
        plt.close()


#List of methods
dir_save = './clustered_corr_matrices/'
dir_save = './clustered_corr_matrices/correlation_hc/'

root_path = './correlation_matrices_hc/'
methods = ['complete', 'single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward']

for method_clustering in methods:
    print(method_clustering)
    repeat_clustering_method(method_clustering, root_path, dir_save)
    