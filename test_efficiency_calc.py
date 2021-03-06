# -*- coding: utf-8 -*-
"""
Testing efficiency by recreating figure http://imaging.mrc-cbu.cam.ac.uk/images/eff_friston.gif on this page https://imaging.mrc-cbu.cam.ac.uk/imaging/DesignEfficiency

@author: Rhodri Cusack
"""

import numpy as np
from single_column_efficiency import efficiency_calc
from matplotlib import pyplot as plt
import pandas as pd
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix

def get_henson_events(ntp=64):
    # Make event prob dist
    des=[]
    des.append((np.arange(ntp)%8==0).astype(np.float))
    des.append(np.ones(ntp)*0.5)
    des.append((1+np.cos(2*np.pi*np.arange(ntp)/8))/2)
    des.append((1+np.cos(2*np.pi*np.arange(ntp)/16))/2)
    des.append((1+np.cos(2*np.pi*np.arange(ntp)/64))/2)
    des.append((np.arange(ntp)<32).astype(np.float))
    ndes = len(des)

    # Make events
    events={}
    
    for axind in range(ndes):
        r=np.random.uniform(size=(ntp,1))
        events[axind]=pd.DataFrame()
        for t in range(ntp):
            if r[t]<des[axind][t]:
                ev={'onset':t,'duration':1,'trial_type':f'model{axind}'}
                events[axind] = events[axind].append(ev, ignore_index=True)
    return events, des


if __name__=='__main__':
    ntp=64
    tr=1.0
    events, des= get_henson_events(ntp=ntp)
    ndes=len(events)

    con=np.zeros((2,1))
    con[0]=1
    eff=[]
    
    X=[]
    fig=plt.figure()
    gs = fig.add_gridspec(ndes, 2)
    for axind in range(ndes):
        ax=fig.add_subplot(gs[axind,0])            
        X=make_first_level_design_matrix(np.arange(ntp) * tr, events=events[axind], hrf_model='spm', drift_model=None)
        ax.bar(range(ntp),des[axind])
        eff.append(efficiency_calc(X,con))
        
    ax=fig.add_subplot(gs[:,1])
    ax.barh(np.arange(0.5,ndes+0.5),eff)
    ax.set_ylim(ndes, 0)
    plt.savefig('test_efficiency.jpg')
    print(eff)


