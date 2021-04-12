import pandas as pd
import matplotlib.pyplot as plt

#manual_F = pd.read_csv('cod_results/stats/longlist_across_movie_Fstats_target_8.csv',index_col=0)
manual_F = pd.read_csv('cod_results/new_longlist/t_stats/model_effiency_select_movies_desmat_blockpermovie_con_all_trial_type_random_weight_target_movies_8.csv', index_col=0)
automatic_F = pd.read_csv('cod_results/stats/all_movie_across_movie_Fstats_target_8.csv',index_col=0)

fig, (ax1,ax2) = plt.subplots(ncols=2, sharex=True, sharey=True)
manual_F.hist(column=['zstat'], ax=ax1)
automatic_F.hist(column=['zstat'], ax=ax2)
ax1.set_title('manually selected')
ax2.set_title('automatically selected')
plt.savefig('./cod_results/new_longlist/Fstat_target_8.jpg')
plt.close()

#manual_t_boiled = pd.read_csv('cod_results/stats/longlist_boiled_down_5_tstats_target_8.csv',index_col=0)
manual_t_boiled = pd.read_csv('cod_results/new_longlist/t_stats/model_effiency_select_movies_desmat_all_con_boiled_down_new_2_target_movies_8.csv',index_col=0)
automatic_t_boiled = pd.read_csv('cod_results/stats/all_movie_boiled_down_5_tstats_target_8.csv',index_col=0)

fig, (ax1,ax2) = plt.subplots(ncols=2, sharex=True, sharey=True)
manual_t_boiled.hist(column=['zstat'], ax=ax1)
automatic_t_boiled.hist(column=['zstat'], ax=ax2)
ax1.set_title('manual (boiled_down_new_2)')
ax2.set_title('automatic (boiled_down_5)')
plt.savefig('./cod_results/new_longlist/tstat_boiled_new_vs_boiled_old_target_8.jpg')
plt.close()

manual_t_neuro = pd.read_csv('cod_results/new_longlist/t_stats/model_effiency_select_movies_desmat_all_con_neuro_based_target_movies_8.csv',index_col=0)
automatic_t_neuro = pd.read_csv('cod_results/stats/all_movie_neuro_based_tstats_target_8.csv',index_col=0)

fig, (ax1,ax2) = plt.subplots(ncols=2, sharex=True, sharey=True)
manual_t_neuro.hist(column=['zstat'], ax=ax1)
automatic_t_neuro.hist(column=['zstat'], ax=ax2)
ax1.set_title('manually selected')
ax2.set_title('automatically selected')
plt.savefig('./cod_results/new_longlist/tstat_neuro_target_8.jpg')
