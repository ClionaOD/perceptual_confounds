"""
Implementation of Bootstrapping MDS Solutions by Jacoby and Armstrong 2013

@author: Cliona O'Doherty
"""

import numpy as np
import pandas as pd
import pickle
import warnings

import scipy.spatial.distance as ssd
from scipy.spatial import procrustes
from sklearn.manifold import MDS

from model_design_matrix import get_design_matrix

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

from contrast_list import get_con_list

warnings.filterwarnings("ignore")


def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', edgecolor='blue', label='', **kwargs):
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
                      facecolor=facecolor, edgecolor=edgecolor, label=label, **kwargs)

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


def construct_rdm(observation_df, metric='correlation'):
    """
    Takes a n*k df with n observations and k conditions,
    returns the data as an n*n rdm dataframe
    """
    rdm = ssd.pdist(observation_df.values.T, metric=metric)
    rdm = ssd.squareform(rdm)
    rdm = pd.DataFrame(rdm, columns=observation_df.columns,
                       index=observation_df.columns)

    return rdm


def get_mds_embedding(rdm, ref=None):
    """
    returns a k*m dataframe with k observations and m dimensions

    ref: a reference mds embedding to which the returned embedding will be aligned (if provided)
    """
    mds = MDS(n_components=2, dissimilarity='precomputed')
    df_embedding = pd.DataFrame(mds.fit_transform(rdm.values), index=rdm.index)

    if ref is not None:
        # Some random combinations of movies may drop a condition (this is rare).
        # Check that both are the same shape, fill in NaN where not.
        if not df_embedding.shape == ref.shape:
            df_embedding = df_embedding.reindex(index=ref.index)
            df_embedding = df_embedding.fillna(0)
        mtx1, mtx2, disparity = procrustes(ref, df_embedding.values)
        df_embedding = pd.DataFrame(mtx2, index=rdm.index)

    return df_embedding


def bootstrapped_mds(events, q=50, con_list=None, reference=None):
    """
    takes events, constructs resampled design matrix, performs bootstrapped mds

    events: dict with keys=movie_names and values=events_files
    q: number of iterations for bootstrapping
    con_list: optional list of contrasts, in form of list of dict/str like this [{'animate':1, 'inanimate':-1}, {'open':1, 'outside':1}, 'social'} where 'social' is equvialent to {'social':1}....]
    reference:    dataframe for Procrustes transform

    """

    # get list of m observations on first iteration of bootstrapping
    observations = []

    bootstrap_embeddings = []

    # All trial types in model, alphabetically
    all_trial_type = list(set().union(
        *[set(events[x].trial_type) for x in events]))
    all_trial_type.sort()

    # Find out how many columns of interest in model
    coi = len(all_trial_type)

    # Make contrast matrix and labels for later
    if not con_list:
        # Default contrasts
        con = np.eye(coi)
        con_names = all_trial_type
    else:
        # Contrasts from list of dict/string (see function description)
        con = np.zeros((coi, len(con_list)))
        con_names = ['']*len(con_list)

        for con_ind, con_list_entry in enumerate(con_list):
            if isinstance(con_list_entry, dict):
                for key, val in con_list_entry.items():
                    con[all_trial_type.index(key), con_ind] = val
                    if val == -1:
                        con_names[con_ind] += '-' + key
                    elif val == 1:
                        con_names[con_ind] += '+' + key
                    else:
                        con_names[con_ind] += f'{val}*{key}'
            elif isinstance(con_list_entry, str):
                con[all_trial_type.index(con_list_entry), con_ind] = 1
                con_names[con_ind] = con_list_entry

    for i in range(q):
        # 1. sample n rows with replacement from V (observation matrix)
        while 1:
            # Repeat if any columns completely missing from design matrix
            bootstrap_replic_q = get_design_matrix(
                events, sample_with_replacement=True)
            if all([x in bootstrap_replic_q for x in all_trial_type]):
                break
        desmat_contrasted = pd.DataFrame(
            np.array(bootstrap_replic_q[all_trial_type]) @ con, columns=con_names)

        # 2. use replication of V to get rdm
        # exclude drift and constant columns
        rdm_q = construct_rdm(desmat_contrasted)

        # 3. perform MDS on the rdm for this iteration
        if i == 0:
            observations.extend(list(rdm_q.columns))
            if reference is None:
                reference = get_mds_embedding(rdm_q)

            mds_q = get_mds_embedding(rdm_q, ref=reference)
        else:
            mds_q = get_mds_embedding(rdm_q, ref=reference)

        bootstrap_embeddings.append(mds_q)

    # 4. restructure the data
    bootstrapped_coords = {}
    for k in con_names:
        # create X, a q*m matrix of bootstrapped coordinates for the condition/object
        X = []
        for mds_q in bootstrap_embeddings:
            x_i = mds_q.loc[k].values
            X.append(x_i)
        bootstrapped_coords[k] = np.array(X)

    return bootstrapped_coords, con_names, reference


def bootstrap_mds_across_movies(con_list=None,
                  events=None,
                  ax=None,
                  use_precomputed_bootstrap=False,
                  use_precomputed_reference=False,
                  nreferences=1,
                  nbootstraps_per_reference=1,
                  ninitial=20,
                  initial_q=50,
                  q=1000,
                  n_std=1.0, 
                  save_results=False,
                  save_figures=False ):

    results = []

    for reference_ind in range(nreferences):
        if use_precomputed_reference:
            with open(f'./bootstrap_mds_reference_coords_q_{q}_nstd_{n_std}.pickle', 'rb') as f:
                reference = pickle.load(f)
        else:
            # Do a set of iterations and find one with lowest variance (so a good reference)
            print('Searching for good reference for MDS')
            reference_min = None
            bootstrap_var_min = np.inf
            for perm in range(ninitial):
                bootstrap_coords, con_names, reference = bootstrapped_mds(
                    events, q=initial_q, con_list=con_list)
                bootstrap_var = 0
                for condition, coords_arr in bootstrap_coords.items():
                    bootstrap_var += coords_arr.var(axis=0).sum()
                if bootstrap_var < bootstrap_var_min:
                    reference_min = reference
                    bootstrap_var_min = bootstrap_var

            reference = reference_min
            print('Found MDS reference')
            with open(f'./bootstrap_mds_reference_coords_q_{q}_nstd_{n_std}.pickle', 'wb') as f:
                pickle.dump(reference, f)

        for bootstrap_ind in range(nbootstraps_per_reference):
            if use_precomputed_bootstrap:
                with open(f'./bootstrap_mds_coords_q_{q}_nstd_{n_std}.pickle', 'rb') as f:
                    bootstrap_coords = pickle.load(f)
            else:
                # Use this reference for the main bootstrap
                bootstrap_coords, con_names, reference = bootstrapped_mds(
                    events, q=q, con_list=con_list, reference=reference)
                with open(f'./bootstrap_mds_coords_q_{q}_nstd_{n_std}.pickle', 'wb') as f:
                    pickle.dump(bootstrap_coords, f)

            cmap = plt.get_cmap('hsv')
            colors = cmap(np.linspace(0, 1.0, len(bootstrap_coords)+1))

            _, ax = plt.subplots(ncols=1, figsize=(11.69, 8.27))

            bootstrap_var = 0
            for idx, (condition, coords_arr) in enumerate(bootstrap_coords.items()):
                x = coords_arr[:, 0]
                y = coords_arr[:, 1]
                # ax.scatter(x,y,s=0.01)
                confidence_ellipse(x, y, ax=ax, n_std=n_std,
                                   edgecolor=colors[idx], label=condition)

                np.mean(coords_arr)

                ax.text(np.mean(x), np.mean(y), condition,
                         ha='center', va='center')
                bootstrap_var += coords_arr.var(axis=0).sum()

            results.append({'reference_ind': reference_ind,
                            'bootstrap_ind': bootstrap_ind,
                            'ninitial': ninitial,
                            'initial_q': initial_q,
                            'q': q,
                            'con_list': con_list,
                            'bootstrap_var': bootstrap_var})
            print(
                f'Reference {reference_ind} iteration {bootstrap_ind} final bootstrap variance {bootstrap_var}')

            ax.set_xlim((-0.35, 0.35))
            ax.set_ylim((-0.35, 0.35))
            ax.set_aspect('equal')
            ax.axis('off')

            h, l = ax.get_legend_handles_labels()
            ax.axis('off')
            legend = ax.legend(h, l,  prop={'size': 5}, loc='upper right')
            plt.tight_layout()

            if save_figures:
                plt.savefig(
                    f'bootstrap/bootstrap_results_q_{q}_std_{n_std}_ninitial_{ninitial}_initialq_{initial_q}_ref_{reference_ind}_iteration_{bootstrap_ind}.pdf')

    if save_results:
        with open(f'bootstrap/bootstrap_results_q{q}_std_{n_std}_ninitial_{ninitial}_initialq_{initial_q}.pickle', 'wb') as f:
            pickle.dump(results, f)

    return results


if __name__ == "__main__":
    events = pd.read_pickle('./events_per_movie.pickle')
    # contrast types [all_trial_type | boiled_down_1| boiled_down_2 | boiled_down_3]
    con_list_type = ['all_trial_type','boiled_down_1','boiled_down_2','boiled_down_3'][0] 

    # Get contrasts
    con_list, nuisance_con, all_trial_type = get_con_list(events, con_list_type, create_duplicate_faces=False)
    
    # No distinction between nuisance and not
    con_list = {**con_list, **nuisance_con}

    results = bootstrap_mds_across_movies(con_list=con_list, events = events, save_results=True, save_figures=True, nreferences=10, nbootstraps_per_reference=2)
