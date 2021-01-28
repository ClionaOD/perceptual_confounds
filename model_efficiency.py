
from nilearn.glm.first_level import FirstLevelModel
from model_design_matrix import get_design_matrix
from bold_noise import bold_noise
import pandas as pd
import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt

from test_efficiency_calc import get_henson_events

def fit_model(events, signal_con, test_con, scale_type='peak2peak', tr=1.0):

    nsamples=1000

    X = get_design_matrix(events, sample_with_replacement=True, tr=tr)
    nscans=len(X)
    print(f'nscans {nscans}')
    bn=bold_noise(nscans=nscans)
    
    subject_data = bn.generate(nsamples=nsamples)

    Y = pd.eval(signal_con) 
    if scale_type=='peak2peak':
        Ymin = Y.min()
        Ymax = Y.max()
        Y =  (Y - Ymin - 1) * (Ymax-Ymin)/2

    Y = Y* 100

    noise_std = subject_data.std(axis=0).mean()
    sig_std = np.std(Y)

    subject_data = subject_data + np.array(Y).reshape(nscans,1)

    print(f'Noise std {noise_std} Signal std {sig_std}')

    img = nib.Nifti1Image(subject_data.T.reshape(nsamples,1,1,nscans), np.eye(4))

    fmri_glm = FirstLevelModel(t_r=tr, hrf_model='spm', mask_img=False)
    fmri_glm = fmri_glm.fit(img, design_matrices=[X])

    z_map = fmri_glm.compute_contrast(test_con, output_type='z_score')
    return z_map

if __name__=='__main__':
    dohenson=True
    tr=2

    if dohenson:
        events, des = get_henson_events()
    else:
        events = pd.read_pickle('./events_per_movie.pickle')
    
    signal_con=0

    all_trial_type=list(set().union(*[set(events[x].trial_type) for x in events]))
    all_trial_type.sort()

    all_vals={}

    
    if dohenson:
        con_list={x:1 for x in all_trial_type} 
    else:
        con_list=['animate',  
            {'biological_motion':1 , 'body_parts': 1},
            {'biological':1, 'social':1},
            'faces', 
            {'inanimate_small':1, 'tools':1},
            'inanimate_big',
            'non_social',  {'salient_near_away':1, 'far':1},
            'salient_near_towards', 
            'near',  
            {'closed':1, 'inside':1}, 
            {'open': 1, 'outside': 1, 'nature': 1},
            'scene', 
            {'non_biological':1, 'civilisation':1},
            'camera_cut',
            {'contrast_sensitivity_function':1, 'global_contrast_factor':1, 'rms_difference':1},
            'scene_change',
        ]

    #all_trial_type = ['open', 'outside', 'nature', ['open','outside','nature']]
    for ind, trial_type in enumerate(con_list):
        if isinstance(trial_type, str):
            trial_type={trial_type: 1}
        sig_con='+'.join([str(val) + '*X.' + key for key, val in trial_type.items()])
        test_con='+'.join([str(val) + '*' + key for key, val in trial_type.items()])
        con_label='+'.join([str(val) + '*' + key[:8] for key, val in trial_type.items()])

        if dohenson:
            ev = {'oneev': events[ind]}
        else:
            ev= events

        vals=fit_model(ev, sig_con, test_con, tr).get_fdata()
        all_vals[con_label] = np.mean(vals)
        
    fig=plt.figure()
    ax = plt.bar(all_vals.keys(), all_vals.values())
    plt.xticks(rotation=90)
    plt.rcParams.update({'font.size': 4})
    plt.tight_layout()
    
    plt.savefig('model_efficiency_bar.jpg')
  