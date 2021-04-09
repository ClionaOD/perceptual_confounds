import pandas as pd
import matplotlib.pyplot as plt

manual_F = pd.read_csv('',index_col=0)
automatic_F = pd.read_csv('cod_results/stats/all_movie_across_movie_Fstats_target_8.csv',index_col=0)

fig, (ax1,ax2) = plt.subplots(ncols=2, sharex=True, sharey=True)
manual_F.hist(column=['zstat'], ax=ax1)
automatic_F.hist(column=['zstat'], ax=ax2)
ax1.set_title('manually selected')
ax2.set_title('automatically selected')
plt.savefig('Fstat_target_8.jpg')
plt.close()

manual_t_boiled = pd.read_csv('cod_results/stats/longlist_boiled_down_5_tstats_target_8.csv',index_col=0)
automatic_t_boiled = pd.read_csv('cod_results/stats/all_movie_boiled_down_5_tstats_target_8.csv',index_col=0)

fig, (ax1,ax2) = plt.subplots(ncols=2, sharex=True, sharey=True)
manual_t_boiled.hist(column=['zstat'], ax=ax1)
automatic_t_boiled.hist(column=['zstat'], ax=ax2)
ax1.set_title('manually selected')
ax2.set_title('automatically selected')
plt.savefig('tstat_boiled_target_8.jpg')
plt.close()

manual_t_neuro = pd.read_csv('',index_col=0)
automatic_t_neuro = pd.read_csv('cod_results/stats/all_movie_neuro_based_tstats_target_8.csv',index_col=0)

fig, (ax1,ax2) = plt.subplots(ncols=2, sharex=True, sharey=True)
manual_t_neuro.hist(column=['zstat'], ax=ax1)
automatic_t_neuro.hist(column=['zstat'], ax=ax2)
ax1.set_title('manually selected')
ax2.set_title('automatically selected')
plt.savefig('tstat_neuro_target_8.jpg')
