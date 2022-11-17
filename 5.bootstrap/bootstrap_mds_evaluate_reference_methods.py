import pandas as pd
from os import path 
import pickle
import numpy as np

tocompare = ['bootstrap_results_q1000_std_1.0_ninitial_1_initialq_1.pickle',]
#           'bootstrap_results_q1000_std_1.0_ninitial_20_initialq_50.pickle']

for datafile in tocompare:
    with open(path.join('bootstrap',datafile),'rb') as f:
        results=pickle.load(f)
    bootstrap_var=np.array([x['bootstrap_var'] for x in results])
    print(f'Mean bootstrap var is {bootstrap_var.mean()} range is {bootstrap_var.min()}-{bootstrap_var.max()}')