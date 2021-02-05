
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

from  joblib import Parallel, delayed

def fit_model(X, events, signal_con, test_con, tr=1.0, sample_with_replacement=True):
    # How many noise samples?
    nsamples = 1000

    nscans = len(X)
#    print(f'nscans {nscans}')

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
#   print(f'Noise std {noise_std} Signal std {sig_std}')

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

def optimise_efficiency(events_in, todrop= None, con_list_type='boiled_down_5', nscans=None, tr=1, save_figures=False, all_trial_type=None):
    
    # Drop anything from events
    if todrop:
        events= events_in.copy()
        del(events[todrop])
    else:
        events= events_in

    scale_type = 'peak2peak' # each column in design matrix

    if all_trial_type is None:
        all_trial_type = list(set().union(*[set(events[x].trial_type) for x in events]))         # List of all trial_type values

    con_list, nuisance_con, _ = get_con_list(events, con_list_type, all_trial_type)

    df = pd.DataFrame()

    # Get design matrix
    X = get_design_matrix(
            events, sample_with_replacement=False, tr=tr, n_scans=nscans, hrf='spm')

    if len(set(all_trial_type) - set(X.columns))>0:
        print('Failing as not all columns in model!')
        return -np.inf


    if scale_type == 'peak2peak':
        # Scale each column in X to have range from 0 to 1
        X[all_trial_type]= (X[all_trial_type] - X[all_trial_type].min()) / (X[all_trial_type].max() - X[all_trial_type].min()) 
    
    if save_figures:
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


    # Revised
    zstat_mean=df['values'].mean()

    if save_figures:
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
#        print(order)
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

    return zstat_mean

if __name__ == '__main__':
    event_type = 'movies'               # event list types
                                        #   henson: use testing set of events from CBU web page (see test_efficiency_calc.py)
                                        #   movies: use all movies
    con_list_type = ['all_trial_type','boiled_down_1','boiled_down_2','boiled_down_3', 'boiled_down_4', 'boiled_down_5'][5] 
    # Get event model.
    if event_type == 'henson':
        original_events, des = get_henson_events()
        nscans = 64
    elif event_type == 'movies':
        original_events = pd.read_pickle('./events_per_movie.pickle')
        nscans = None # calculated from events

    res = pd.DataFrame()

    all_trial_type = list(set().union(*[set(original_events[x].trial_type) for x in original_events]))         # List of all trial_type values

    
    num_movies=len(original_events)
    zstat = optimise_efficiency(original_events)
    print(f'All movies {zstat}')

    finals=[]

    nits=100

    target_movies = 8
    
    # for testing, original_events = {k: original_events[k] for k in ['walle.mp4', 'cars.mp4', 'up_russell.mp4', 'real_cars.mp4', 'breakfast_smoothie.mp4', 'new_orleans.mp4', 'funny_tools.mp4', 'despicable_me.mp4', 'steamed_buns.mp4']}

    # Iteratively drop movies that give the lowest average z stat
    for it in range(nits):
        # Make a copy before we start removing movies
        events=original_events.copy()
        for ind in range(num_movies, target_movies - 1, -1):
            keys = list(events.keys())
            zstat = Parallel(n_jobs=16)(delayed(optimise_efficiency)(events, todrop=leftout, all_trial_type=all_trial_type) for leftout in keys)
#            zstat =[optimise_efficiency(events, todrop=leftout, all_trial_type=all_trial_type) for leftout in keys]
            index_max = max(range(len(zstat)), key=zstat.__getitem__) 
            print(f'Dropping {keys[index_max]} as still gave z-stat of {zstat[index_max]}')
            del events[keys[index_max]]
        res = res.append({'iteration':it, 'movies':keys}, ignore_index=True)
        print(f'Iteration {it} final set is {keys}' )

    with open('model_effiency_select_movies.csv','w') as f:
        res.to_csv(f)

    # ['rio_jungle_jig.mp4', 'bedtime.mp4', 'up_russell.mp4', 'ratatouille.mp4', 'real_cars.mp4', 'supermarket.mp4', 'funny_tools.mp4', 'dalmations.mp4', 'beachsong.mp4']
 #  ['walle.mp4', 'rio_jungle_jig.mp4', 'forest.mp4', 'cars.mp4', 'ratatouille.mp4', 'bathsong.mp4', 'up_kevin_nonsocial.mp4', 'supermarket.mp4', 'funny_tools.mp4']
