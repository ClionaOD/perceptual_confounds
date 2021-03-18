import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning)
    from nilearn.glm.first_level import FirstLevelModel
    from nilearn.plotting import plot_design_matrix
from model_design_matrix import get_design_matrix, get_df_events
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

def fit_model(X, signal_con, test_con, tr=1.0, Xtest = None, stat_type = 't'):
    
    if Xtest is None:
        Xtest = X

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
    fmri_glm = fmri_glm.fit(img, design_matrices=[Xtest])

    z_map = fmri_glm.compute_contrast(test_con, output_type='z_score', stat_type = stat_type)
    return z_map, Y


def get_multiplier(val):
    # Used when contsructing formula for pandas.eval
    if val == 1:
        return ''
    elif val == -1:
        return '-'
    else:
        return '%.2f*'%val


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

def optimise_efficiency(events_in, todrop= None, con_list_type='boiled_down_5', nscans=None, tr=1, save_figures=False, all_trial_type=None, design_matrix_type = 'all', target_movies=8 ):
    '''
    Parameters
        events_in: dict, with keys of movie name, and values a pandas dataframe describing the design
        todrop: string, a movie to be dropped from events_in (used to make leave-one-out more convenient to parallelise)
        con_list_type: string, see possible values in contrast_list.py
        nscans: number of scans in design matrix, or None if calculate from events
        tr: repetition time
        save_figures: save outputs
        event_type: 'movies', just used to label output files
        design_matrix_type: ['all' | 'single'] 
            'all' means put all columns into model no matter what contrast signal is
            'percontrast' means only put contrast signal
    Returns
        zstat: estimated z statistic
    '''

    event_type='movies'
    movie_length = 22.524
    delay = 23.0 - movie_length

    if not design_matrix_type == 'all':
        event_type = event_type + '_desmat_percontrast'

    # Drop anything requested from events 
    if todrop:
        events= events_in.copy()
        del(events[todrop])
    else:
        events= events_in

    scale_type = 'peak2peak' # each column in design matrix

    # List of all events
    if all_trial_type is None:
        all_trial_type = list(set().union(*[set(events[x].trial_type) for x in events]))         # List of all trial_type values

    # Signal and nuisance contrasts
    con_list, nuisance_con, _ = get_con_list(events, con_list_type, all_trial_type)

    df = pd.DataFrame()

    # Get design matrix
    #  Used to generate signal, and if design_matrix_type = 'all', for test as well
    stacked_events, n_scans, list_videos = get_df_events(events, rest_length=0.0, sample_with_replacement=False, n_scans=None, movie_length=movie_length, delay=delay)
    X = get_design_matrix(tr=tr, hrf='spm', stacked_events = stacked_events, n_scans = n_scans)

    if design_matrix_type=='blockpermovie':
        # Create design matrix for movies
        movie_events = pd.DataFrame({'onset': np.arange(len(events)) * (movie_length+delay), 'duration': movie_length, 'trial_type': list_videos})
        Xmovies = get_design_matrix(tr=tr, hrf='spm', stacked_events = movie_events, n_scans = n_scans)
        if save_figures:
            plt.figure()
            plot_design_matrix(Xmovies)
            plt.savefig('./cod_results/model_efficiency_events_blockpermovie_designmatrix.jpg')

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
        plt.savefig(f'./cod_results/model_efficiency_events_Xcorr.jpg')

    # Contrasted design matrix columns
    Xcon=pd.DataFrame()

    for ind, trial_type in enumerate(con_list):
        if isinstance(trial_type, str):
            trial_type = {trial_type: 1}

        # These columns are used to make the specification for the simulated brain signal, in sig_con.
        #   Includes nuisance_con, with random weighting from 0 to the contrast value
        if trial_type is None:
            sig_con=''
            con_label = 'F-test'
        else:
            sig_con = '+'.join([get_multiplier(val) + 'X.' +
                                key for key, val in trial_type.items()])
            # Contrast for which result should be returned
            #  This is the thing we measure
            test_con = '+'.join([get_multiplier(val) +
                                key for key, val in {**trial_type}.items()])
            # Used as labels in figures
            #  A short form for this contrast, to use in figures
            con_label = '+'.join([get_multiplier(val) + get_abbrev(key)
                                for key, val in trial_type.items()])
        if nuisance_con:
            sig_con += '+' + '+'.join([get_multiplier(np.random.uniform(
                high=float(val))) + 'X.' + key for key, val in nuisance_con.items()])




        # Find design matrix type to use 
        if design_matrix_type == 'percontrast':
            # Select only events of trial_type in test_con
            test_stacked_events = pd.DataFrame()
            for tt in trial_type:
                test_stacked_events = test_stacked_events.append(stacked_events[stacked_events.trial_type == tt])

            # Make limited design matrix using these. Use same order for events
            Xtest = get_design_matrix(stacked_events = test_stacked_events, sample_with_replacement=False, tr=tr, n_scans=n_scans, hrf='spm')
            if scale_type == 'peak2peak':
                # Scale each non-nuisance column in Xtest to have range from 0 to 1
                ttk=list(trial_type.keys())
                Xtest[ttk]= (Xtest[ttk] - Xtest[ttk].min()) / (Xtest[ttk].max() - Xtest[ttk].min()) 

        elif design_matrix_type == 'all':
            Xtest = X
        elif design_matrix_type == 'blockpermovie':
            Xtest = Xmovies
        else:
            raise(f'Unknown design_matrix_type {design_matrix_type}')

        # Fit GLM
        #  sample_with_replacement determines whether to do bootstrapping across movies
        if design_matrix_type == 'blockpermovie':
            # leave off a movie, with np.eye(#movies - 1) so that contrast for F test is not (nearly) singluar due to constant columns
            zstats, concol = fit_model(X, sig_con, np.eye(M= len(Xtest.columns), N = len(events)-1), Xtest = Xtest, tr=tr, stat_type='F')
        else:
            zstats, concol = fit_model(X, sig_con, test_con, Xtest = Xtest, tr=tr)

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
        plt.plot([-0.5,len(con_list)], [0,0], 'r:')
        plt.xticks(rotation=90)
        plt.ylabel('z stat')
        plt.rcParams.update({'font.size': 12})
        plt.tight_layout()
    
        plt.savefig(f'./cod_results/model_efficiency_events_{event_type}_con_{con_list_type}_violin.jpg')

        fig, ax = plt.subplots(nrows = 1, figsize=(11.5, 9))
        
        if len(con_list)>1:
            # Only if not F-test
            #Correlation matrix
            corrMatrix = Xcon.corr()

            # Clustering figure
            dend = sch.dendrogram(sch.linkage(corrMatrix,  method= 'complete'), 
                    ax=ax,  
                    labels=corrMatrix.columns,
                    orientation='right'
                )
            order=dend['ivl']

            plt.savefig(f'./cod_results/model_efficiency_events_{event_type}_con_{con_list_type}_clusters.jpg')

            # Correlation matrix figure
            fig, ax = plt.subplots(figsize=(11.5, 9))
            plt.rcParams.update({'font.size': 12})
            sns.heatmap(corrMatrix, ax=ax, cmap='PiYG', vmin=-0.5, vmax=0.5)
            plt.tight_layout()
            plt.savefig(f'./cod_results/model_efficiency_events_{event_type}_con_{con_list_type}_corr.jpg')

            # Correlation matrix, reordered by clustering, figure
            corrMatrixReordered = corrMatrix.reindex(index=order,columns=order)
            fig, ax = plt.subplots(figsize=(11.5, 9))
            plt.rcParams.update({'font.size': 12})
            sns.heatmap(corrMatrixReordered, ax=ax, cmap='PiYG', vmin=-0.5, vmax=0.5)
            plt.tight_layout()
            plt.savefig(f'./cod_results/model_efficiency_events_{event_type}_con_{con_list_type}_corr_reordered.jpg')

    return zstat_mean

def find_optimal_movies(nits=1, target_movies=8, design_matrix_type='all', con_list_type='boiled_down_5'):
    original_events = pd.read_pickle('./events_per_movie_longlist.pickle')
    nscans = None # calculated from events

    res = pd.DataFrame()

    all_trial_type = list(set().union(*[set(original_events[x].trial_type) for x in original_events]))         # List of all trial_type values
    
    num_movies=len(original_events)
    zstat = optimise_efficiency(original_events, all_trial_type=all_trial_type, design_matrix_type=design_matrix_type, con_list_type=con_list_type)
    print(f'All movies {zstat}')

    finals=[]

    # for testing, original_events = {k: original_events[k] for k in ['walle.mp4', 'cars.mp4', 'up_russell.mp4', 'real_cars.mp4', 'breakfast_smoothie.mp4', 'new_orleans.mp4', 'funny_tools.mp4', 'despicable_me.mp4', 'steamed_buns.mp4']}

    # Iteratively drop movies that give the lowest average z stat
    for it in range(nits):
        # Make a copy before we start removing movies
        events=original_events.copy()
        for ind in range(num_movies, target_movies - 1, -1):
            keys = list(events.keys())
            zstat = Parallel(n_jobs=16)(delayed(optimise_efficiency)(events, todrop=leftout, all_trial_type=all_trial_type, design_matrix_type=design_matrix_type, con_list_type=con_list_type) for leftout in keys)
#            zstat =[optimise_efficiency(events, todrop=leftout, all_trial_type=all_trial_type) for leftout in keys]
            index_max = max(range(len(zstat)), key=zstat.__getitem__) 
            print(f'Dropping {keys[index_max]} as still gave z-stat of {zstat[index_max]}')
            del events[keys[index_max]]


        zstat = optimise_efficiency(events, all_trial_type=all_trial_type, design_matrix_type=design_matrix_type, con_list_type=con_list_type) 
        res = res.append({'iteration':it, 'movies':keys, 'zstat':zstat}, ignore_index=True)
        print(f'Iteration {it} final set is {keys}' )

    with open(f'./cod_results/t_stats/model_effiency_select_movies_desmat_{design_matrix_type}_con_{con_list_type}_target_movies_{target_movies}.csv','w') as f:
        res.to_csv(f)


def all_movie_analysis(con_list_type= 'boiled_down_5', event_type = 'movies', design_matrix_type='all', target_movies=8):
    # Get event model.
    if event_type == 'henson':
        events, des = get_henson_events()
        nscans = 64
    elif event_type == 'movies':
        events = pd.read_pickle('./events_per_movie_longlist.pickle')
        nscans = None # calculated from events


    zstat = optimise_efficiency(events, save_figures=True, con_list_type = con_list_type, design_matrix_type=design_matrix_type, target_movies=8)
    print(f'All movies {zstat}')

if __name__=='__main__':
    # Find best subset of 8 movies, given contrasts 
    # find_optimal_movies()

    # Run once with figures
    #all_movie_analysis(con_list_type = 'all_trial_type', design_matrix_type = 'percontrast')
    #all_movie_analysis(con_list_type = 'boiled_down_5', design_matrix_type = 'all')
    #all_movie_analysis(con_list_type = 'neuro_based', design_matrix_type = 'all')
    
    # Get t stat distributions for all movies in both cluster-based contrast and neuro-based
    #find_optimal_movies(con_list_type = 'boiled_down_5', design_matrix_type = 'all', nits=100, target_movies=15)
    #find_optimal_movies(con_list_type = 'neuro_based', design_matrix_type = 'all', nits=100, target_movies=15)
    
    find_optimal_movies(con_list_type = 'boiled_down_5', design_matrix_type = 'all', nits=100, target_movies=8)
    find_optimal_movies(con_list_type = 'neuro_based', design_matrix_type = 'all', nits=100, target_movies=8)
    
    # Assume each voxel of simulated brain activating with random weights on each tagged column, but that we're analysing for differences across movies
    #  We do an F-test for the effect of movie
    #  Then optimise subset of movies to maxmise this
    find_optimal_movies(con_list_type = 'all_trial_type_random_weight', design_matrix_type = 'blockpermovie', nits=100, target_movies=8)

    # *** NEXT STAGE ***    
    # Can use script summarize_select_movies.py to summarize results - change this to pick up correct .csv output files
