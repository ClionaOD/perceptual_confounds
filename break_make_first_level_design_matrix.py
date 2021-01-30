import pickle
import pandas as pd
from nilearn.glm.first_level import make_first_level_design_matrix
import numpy as np
from matplotlib import pyplot as plt

frame_times=np.arange(20)

fig, ax= plt.subplots(nrows=2)

# All good
events=pd.DataFrame({'onset': [2, 4, 4.5], 'duration': [1.0, 1.0, 1.0], 'trial_type': ['vanilla']*3})
X = make_first_level_design_matrix(frame_times, events= events, hrf_model=None)
ax[0].plot(X['vanilla'])

# Second and third events end at same time, so first event never ends!
events=pd.DataFrame({'onset': [2, 4, 4.5], 'duration': [1.0, 1.5, 1.0], 'trial_type': ['vanilla']*3})
X = make_first_level_design_matrix(frame_times, events= events, hrf_model=None)
ax[1].plot(X['vanilla'])

plt.savefig('break_make_first_level_design_matrix.jpg')
