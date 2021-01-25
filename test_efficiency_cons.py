# -*- coding: utf-8 -*-
"""
Trying to work out why sums of columns do funny things to efficiency

@author: Rhodri Cusack
"""
import numpy as np
from single_column_efficiency import efficiency_calc
from matplotlib import pyplot as plt
import pandas as pd
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix


# Make event prob dist
ntp=64
des=[]
#des.append((np.arange(ntp)==16).astype(np.float)) # wimpy column!
des.append(0.9*(np.arange(ntp)<32).astype(np.float)) # really powerful column!
des.append(0.9*(np.arange(ntp)<32).astype(np.float)) # really powerful column but similar!
ndes = len(des)

# Make events
events={}
X=[]
fig=plt.figure()
gs = fig.add_gridspec(ndes, 2)
events=pd.DataFrame()

for axind in range(ndes):
    ax=fig.add_subplot(gs[axind,0])
    r=np.random.uniform(size=(ntp,1))
    for t in range(ntp):
        if r[t]<des[axind][t]:
#            ev={'onset':t + axind*0.0001,'duration':1,'trial_type':f'cond{axind}'} # jittering onsets with axind * 0.0001 due to behaviour of make_first_level_design_matrix raised in this issue https://github.com/nilearn/nilearn/issues/2668#issuecomment-766424736
            ev={'onset':t + np.random.uniform(0.5),'duration':1,'trial_type':f'cond{axind}'} # adding some random jitter to the columns
            events = events.append(ev, ignore_index=True)
all_trial_type=['cond0','cond1']

X=make_first_level_design_matrix(np.arange(ntp), events=events, hrf_model='spm', drift_model=None)

events['trial_type']='cond_both'
Xboth=make_first_level_design_matrix(np.arange(ntp), events=events, hrf_model='spm', drift_model=None)
print(events)
X[all_trial_type]= (X[all_trial_type] - X[all_trial_type].mean()) / X[all_trial_type].std()
Xboth['cond_both']= (Xboth['cond_both'] - Xboth['cond_both'].mean()) / Xboth['cond_both'].std()


print(X)

# Now contrast - first col, second col, both together
con=np.zeros((3,1))
con[:2,0]=[1,0]
print(efficiency_calc(X,con))

con[:2,0]=[0,1]
print(efficiency_calc(X,con))

con=np.zeros((3,1))
con[:2,0]=[1,1]
print(efficiency_calc(X,con))

fig,ax = plt.subplots(nrows=2)

ax[0].plot(X)
# And on regressor with events combined
con=np.zeros((2,1))
con[0]=[1]
print(efficiency_calc(Xboth,con))
ax[1].plot(Xboth)

plt.savefig('test_efficiency_cons_desmat.jpg')


