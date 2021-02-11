"""
Implementation of Bootstrapping MDS Solutions by Jacoby and Armstrong 2013

@author: Cliona O'Doherty
"""

import numpy as np
import pandas as pd
import pickle

import scipy.spatial.distance as ssd
from scipy.spatial import procrustes
from sklearn.manifold import MDS

from model_design_matrix import get_design_matrix

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

#Steps
#1. Bootstrap the samples
#2. Calculate rdm: pairwise distances
#3. Create Groupings from rdm
#4. 

def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
    """
    Copied from https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html
    
    ----------
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, edgecolor='blue', **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    
    return ax.add_patch(ellipse)

def construct_rdm(observation_df):
    """
    Takes a n*k df with n observations and k conditions,
    returns the data as an n*n rdm dataframe #? Not k*k
    """
    rdm = ssd.pdist(observation_df.values.T) #*? Why the Transpose 
    rdm = ssd.squareform(rdm)
    rdm = pd.DataFrame(rdm, columns=observation_df.columns, index=observation_df.columns)
    
    return rdm

def get_mds_embedding(rdm, ref=None):
    """
    returns a k*m dataframe with k observations and m dimensions

    ref: a reference mds embedding to which the returned embedding will be aligned (if provided)
    """
    mds = MDS(n_components=2, dissimilarity='precomputed')
    df_embedding = pd.DataFrame(mds.fit_transform(rdm.values), index=rdm.index)
    if ref is not None:
        mtx1, mtx2, disparity = procrustes(ref, df_embedding.values) #? Question
        df_embedding = pd.DataFrame(mtx2, index=rdm.index)
    
    return df_embedding

def bootstrapped_mds(events, q=50):
    """
    takes events, constructs resampled design matrix, performs bootstrapped mds
    
    events: dict with keys=movie_names and values=events_files
    q: number of iterations for bootstrapping
    """
    
    # get list of m observations on first iteration of bootstrapping
    observations = []
    
    bootstrap_embeddings = []
    for i in range(q):
        # 1. sample n rows with replacement from V (observation matrix)
        bootstrap_replic_q = get_design_matrix(events, sample_with_replacement=True)
        
        #if sample_with_replacement:
        #list_videos = random.choices(list_videos,k=len(list_videos))
        #else:
        #random.shuffle(list_videos) #Randomise movie order
        #df = get_events_df(list_videos)

        # 2. use replication of V to get rdm
        rdm_q = construct_rdm(bootstrap_replic_q.iloc[:,:-14]) #exclude drift and constant. Columns nxn df of conditions

        # 3. perform MDS on the rdm for this iteration
        if i == 0:
            observations.extend(list(rdm_q.columns))
            mds_q = get_mds_embedding(rdm_q)
        else:
            mds_q = get_mds_embedding(rdm_q, ref=bootstrap_embeddings[0])
        
        bootstrap_embeddings.append(mds_q)

    # 4. restructure the data
    bootstrapped_coords = {}
    for k in observations:
        # create X, a q*m matrix of bootstrapped coordinates for the condition/object
        X = []
        for mds_q in bootstrap_embeddings:
            x_i = mds_q.loc[k].values #Returns each row as a series --> get values 
            X.append(x_i)
        bootstrapped_coords[k] = np.array(X) #x coord, y coord 

    return bootstrapped_coords

if __name__ == "__main__":
    events = pd.read_pickle('./events_per_movie.pickle')

    bootstrap_coords = bootstrapped_mds(events, q=50)

    fig, ax = plt.subplots()
    for condition, coords_arr in bootstrap_coords.items():
        x = coords_arr[:,0].T ; y = coords_arr[:,1].T
        ax.scatter(x,y,s=0.5)
        confidence_ellipse(x, y, ax=ax, facecolor='red')
    plt.savefig('bootstrap_mds.jpg')


