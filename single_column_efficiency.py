
from model_design_matrix import get_design_matrix
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from nilearn.plotting import plot_design_matrix


def efficiency_calc(X, contrasts):
    '''Calculate efficiency for a given design matrix '''     
    #Singular matrix - no solution to inverse 
    invXtX = np.linalg.pinv(X.T@X)
    efficiency = 1.0 / np.trace(( contrasts.T@invXtX@contrasts))
    return efficiency

if __name__=='__main__':
    events = pd.read_pickle('./events_per_movie.pickle')

    all_trial_type=list(set().union(*[set(events[x].trial_type) for x in events]))
    all_trial_type.sort()

    X=get_design_matrix(events, sample_with_replacement=True)


    num_coi = len(all_trial_type)  # columns of interest
    num_nuisance = X.shape[1] - num_coi

    # standardise columns of interest
    X[all_trial_type] = (X[all_trial_type] - X[all_trial_type].mean()) / X[all_trial_type].std(ddof=0)

    plot_design_matrix(X,output_file='desmat.jpg')

    # Efficiency for each column, when accompanied by only the nuisance regressors
    eff={}
    for trial_type in all_trial_type:
        Xtrim = X[[trial_type]+ list(X.columns[-num_nuisance:])]
        con = np.zeros((num_nuisance + 1,1))
        con[0,0] = 1
        eff[trial_type]=efficiency_calc(Xtrim, con)

    # Graph up
    fig, ax= plt.subplots()
    plt.bar(range(len(eff)), list(eff.values()), align='center')
    plt.xticks(range(len(eff)), list(eff.keys()), rotation=90)


    # Efficiency for each column, when accompanied by all other regressors
    eff_withall={}
    for trial_type in all_trial_type:
        con = np.zeros((num_coi + num_nuisance,1))
        con[X.columns == trial_type,0] = 1
        eff_withall[trial_type]=efficiency_calc(X, con)

    # Graph up
    plt.bar(range(len(eff_withall)), list(eff_withall.values()), align='center')
    plt.xticks(range(len(eff_withall)), list(eff_withall.keys()), rotation=90)
    fig.tight_layout()
    plt.savefig('single_efficiency.jpg')

    # Bespoke contrasts
    cons_text=[['outside'], ['open'], ['nature'], ['outside', 'open', 'nature']]
    eff_bespoke={}
    for con_text in cons_text:
        con = np.zeros((num_coi + num_nuisance,1))
        for item in con_text:
            con[X.columns == item, 0] = 1
        eff_bespoke['+'.join(con_text)]=efficiency_calc(X, con)

    # Probing what is going on
    fig, ax = plt.subplots()
    plt.plot('outside', data=X)
    plt.plot('nature', data=X)
    plt.plot('open', data=X)
    plt.legend()
    plt.savefig('bespoke_timecourse.jpg')

    # Graph up
    fig,ax = plt.subplots()
    plt.bar(range(len(eff_bespoke)), list(eff_bespoke.values()), align='center')
    plt.xticks(range(len(eff_bespoke)), list(eff_bespoke.keys()), rotation=90)
    plt.tight_layout()
    plt.savefig('bespoke_efficiency.jpg')

