from joblib import Parallel, delayed
from math import sqrt

a=Parallel(n_jobs=8)(delayed(sqrt)(i) for i in range(10))
print(a)