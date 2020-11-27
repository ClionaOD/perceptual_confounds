import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from get_dend import hierarchical_clustering

gcf_df = pd.read_csv('./mean_gcf.csv', index_col=0)
gcf_df.columns=['mean','std']

gcf_rdm = pd.DataFrame(index=gcf_df.index, columns=gcf_df.index, dtype=float)

for movie1, results1 in gcf_df.iterrows():
    for movie2, results2 in gcf_df.iterrows():
        mean1 = results1['mean']
        mean2 = results2['mean']
        
        gcf_rdm[movie1][movie2] = float(abs(mean1-mean2))

cluster_order = hierarchical_clustering(gcf_rdm.values, gcf_rdm.index)

reord_df = gcf_rdm.reindex(index=cluster_order,columns=cluster_order)

fig,ax = plt.subplots(figsize=(11.69,8.27))
sns.heatmap(reord_df, ax=ax)
ax.tick_params(axis='both',labelsize=9)
plt.tight_layout
plt.savefig('mean global contrast RDM.pdf')
plt.show()
plt.close()