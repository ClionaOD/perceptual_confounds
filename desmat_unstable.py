import pickle
import pandas as pd
from model_design_matrix import get_design_matrix
import numpy as np
from matplotlib import pyplot as plt


with open('./events_per_movie.pickle','rb') as f:
    events = pickle.load(f)

colname='salient_near_towards'

for vid in events:
    events[vid] = events[vid][events[vid]['trial_type']==colname]
    print(events[vid])

nxrep=10
allX = np.zeros((690, nxrep))
xmean = []
fig, ax = plt.subplots(nrows=nxrep)
for Xrep in range(nxrep):
    X = get_design_matrix(events, sample_with_replacement=False, tr=1, n_scans=690, hrf=None)
    xmean.append(X[colname].mean())
    allX[:,Xrep]=np.array(X[colname])
    ax[Xrep].plot(X[colname])

#plt.imshow(allX.T, aspect='auto', interpolation=None)

plt.savefig('desmat_unstable.jpg')


