
from nilearn.glm.first_level import make_first_level_design_matrix
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


data = [[0, 1, 'cond1'], [0, 1, 'cond1'], [10, 1, 'cond1']] # duplicate event at 0 secs not counted
events = pd.DataFrame(data, columns=['onset','duration','trial_type'])
X0 = make_first_level_design_matrix(np.arange(20),events=events)

print(events.duplicated().any())

data = [[0, 1, 'cond1'], [0.001, 1, 'cond1'], [10, 1, 'cond1']] # duplicate event ~ 0 secs is now counted
events = pd.DataFrame(data, columns=['onset','duration','trial_type'])
X1 = make_first_level_design_matrix(np.arange(20),events=events)

fig, ax = plt.subplots(nrows=2)
ax[0].plot(X0)
ax[1].plot(X1)
plt.savefig('show_make_first_level_design_matrix_oddity.jpg')
