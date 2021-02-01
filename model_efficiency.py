
from nilearn.glm.first_level import FirstLevelModel
from model_design_matrix import get_design_matrix
from bold_noise import bold_noise
import pandas as pd
import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from test_efficiency_calc import get_henson_events
import scipy.cluster.hierarchy as sch

from contrast_list import get_con_list

def fit_model(X, events, signal_con, test_con, tr=1.0, sample_with_replacement=True):
    # How many noise samples?
    nsamples = 1000

    nscans = len(X)
    print(f'nscans {nscans}')

    # Generate BOLD noise
    bn = bold_noise(nscans=nscans)
    subject_data = bn.generate(nsamples=nsamples)

    # Create simulated brain signal
    Y = pd.eval(signal_con)

    # Scale
    Y = Y * 100

    # Noise and signal stats
    noise_std = subject_data.std(axis=0).mean()
    sig_std = np.std(Y)
    print(f'Noise std {noise_std} Signal std {sig_std}')

    # Add signal to noise and place into Nifti 
    subject_data = subject_data + np.array(Y).reshape(nscans, 1)
    img = nib.Nifti1Image(subject_data.T.reshape(
        nsamples, 1, 1, nscans), np.eye(4))

    # Fit GLM
    fmri_glm = FirstLevelModel(t_r=tr, hrf_model='spm', mask_img=False)
    fmri_glm = fmri_glm.fit(img, design_matrices=[X])

    z_map = fmri_glm.compute_contrast(test_con, output_type='z_score')
    return z_map, Y


def get_multiplier(val):
    # Used when contsructing formula for pandas.eval
    if val == 1:
        return ''
    elif val == -1:
        return '-'
    else:
        return f'{val}*'


def get_abbrev(val):
    # Get abbreviated label for figures
    abbrevs = {'contrast_sensitivity_function': 'csf',
               'global_contrast_factor': 'gcf',
               'rms_difference': 'rms',
               'biological': 'bio',
               'salient_near_away': 'sal_away',
               'salient_near_towards': 'sal_tow',
               'animate': 'anim',
               'outside': 'out',
               'inside': 'in',
    }
    for k,v in abbrevs.items():
        val = val.replace(k,v)
    return val

if __name__ == '__main__':
    tr = 1
    # create nuisance variable that is approximate duplicate of faces
    create_duplicate_faces = False
    event_type = 'movies'               # event list types
                                        #   henson: use testing set of events from CBU web page (see test_efficiency_calc.py)
                                        #   movies: use all movies
    con_list_type = ['all_trial_type','boiled_down_1','boiled_down_2','boiled_down_3'][0] 
                                        
    
    scale_type = 'peak2peak' # each column in design matrix

    # Get event model.
    if event_type == 'henson':
        events, des = get_henson_events()
        nscans = 64
    elif event_type == 'movies':
        events = pd.read_pickle('./events_per_movie.pickle')
        nscans = None # calculated from events

    con_list, nuisance_con, all_trial_type = get_con_list(events, con_list_type, create_duplicate_faces=False)

    #all_trial_type = ['open', 'outside', 'nature', ['open','outside','nature']]
    df = pd.DataFrame()

    # Get design matrix
    X = get_design_matrix(
        events, sample_with_replacement=False, tr=tr, n_scans=nscans, hrf='spm')

    if scale_type == 'peak2peak':
        # Scale each column in X to have range from 0 to 1
        all_trial_type = list(set().union(*[set(events[x].trial_type) for x in events]))         # List of all trial_type values
        X[all_trial_type]= (X[all_trial_type] - X[all_trial_type].min()) / (X[all_trial_type].max() - X[all_trial_type].min()) 

    # Dump correlation
    corrX = X.corr()
    fig, ax = plt.subplots(figsize=(11.5, 9))
    plt.rcParams.update({'font.size': 8})
    sns.heatmap(corrX, ax=ax, cmap='PiYG', vmin=-0.5, vmax=0.5)
    plt.tight_layout()

    plt.savefig(f'model_efficiency_events_Xcorr.jpg')

    # contrasted design matrix columns
    Xcon=pd.DataFrame()

    for ind, trial_type in enumerate(con_list):
        if isinstance(trial_type, str):
            trial_type = {trial_type: 1}

        # These columns are used to make the simulated brain signal. Includes nuisance_con, with random weighting.
        sig_con = '+'.join([get_multiplier(val) + 'X.' +
                            key for key, val in trial_type.items()])
        if nuisance_con:
            sig_con += '+' + '+'.join([get_multiplier(np.random.uniform(
                high=float(val))) + 'X.' + key for key, val in nuisance_con.items()])

        # Contrast for which result should be returned
        test_con = '+'.join([get_multiplier(val) +
                             key for key, val in {**trial_type}.items()])

        # Used as labels in figures
        con_label = '+'.join([get_multiplier(val) + get_abbrev(key)
                              for key, val in trial_type.items()])

        if event_type == 'henson':
            ev = {'oneev': events[ind]}
        else:
            ev = events

        # Fit GLM
        #  sample_with_replacement determines whether to do bootstrapping across movies
        zstats, concol = fit_model(X, ev, sig_con, test_con, tr=tr, sample_with_replacement=False)

        vals = zstats.get_fdata()

        Xcon[con_label]=concol

        # Store result
        all_vals = {'con_label': [con_label]*len(vals), 'values': vals.ravel()}
        df = df.append(pd.DataFrame(all_vals), ignore_index=True)

    # Violin plot of zstats for each contrast
    fig, ax = plt.subplots(nrows = 1, figsize=(11.5, 9))
    plt.grid(axis='x')
    sns.violinplot(data=df, x='con_label', y='values', ax=ax)
    plt.xticks(rotation=90)
    plt.ylabel('z stat')
    plt.rcParams.update({'font.size': 12})
    plt.tight_layout()
  
    plt.savefig(f'model_efficiency_events_{event_type}_con_{con_list_type}_violin.jpg')

    fig, ax = plt.subplots(nrows = 1, figsize=(11.5, 9))
    #Correlation matrix
    corrMatrix = Xcon.corr()

    # Clustering figure
    dend = sch.dendrogram(sch.linkage(corrMatrix,  method= 'complete'), 
            ax=ax,  
            labels=corrMatrix.columns,
            orientation='right'
        )
    order=dend['ivl']
    print(order)
    plt.savefig(f'model_efficiency_events_{event_type}_con_{con_list_type}_clusters.jpg')

    # Correlation matrix figure
    fig, ax = plt.subplots(figsize=(11.5, 9))
    plt.rcParams.update({'font.size': 12})
    sns.heatmap(corrMatrix, ax=ax, cmap='PiYG', vmin=-0.5, vmax=0.5)
    plt.tight_layout()
    plt.savefig(f'model_efficiency_events_{event_type}_con_{con_list_type}_corr.jpg')

    # Correlation matrix, reordered by clustering, figure
    corrMatrixReordered = corrMatrix.reindex(index=order,columns=order)
    fig, ax = plt.subplots(figsize=(11.5, 9))
    plt.rcParams.update({'font.size': 12})
    sns.heatmap(corrMatrixReordered, ax=ax, cmap='PiYG', vmin=-0.5, vmax=0.5)
    plt.tight_layout()
    plt.savefig(f'model_efficiency_events_{event_type}_con_{con_list_type}_corr_reordered.jpg')
