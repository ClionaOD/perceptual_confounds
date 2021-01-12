import pandas as pd
import numpy as np

#turn framewise global contrast function results into an events dataframe, one per movie
gcf_df = pd.read_csv('./framewise_gcf.csv', index_col=0)
gcf_events = {k:None for k in gcf_df.index}

for idx, vid in enumerate(gcf_df.index):
    df = pd.DataFrame(columns=['onset','duration','trial_type','magnitude'])
    #set onset to be from 0 to nframes in intervals of fps
    df['onset'] = np.arange(0, (1/25)*563, 1/25)
    #set duration to be framerate in sec i.e. 1/25 sec
    df['duration'] = [1/25 for i in range(563)]
    #set trial_type as this confound, gcf
    df['trial_type'] = ['global_contrast_factor' for i in range(563)]
    #magnitude is the gcf value
    df['magnitude'] = gcf_df.loc[vid].values

    gcf_events[vid] = df

csf_df = pd.read_csv('./total_energy.csv')
csf_events = {k:None for k in csf_df.columns}
for idx, vid in enumerate(csf_df.columns):
    df = pd.DataFrame(columns=['onset','duration','trial_type','magnitude'])
    
    #set onset to be from 0 to nframes in intervals of fps
    df['onset'] = list(range(21))
    #set duration to be framerate in sec i.e. 1/25 sec
    df['duration'] = [1 for i in range(21)]
    #set trial_type as this confound, gcf
    df['trial_type'] = ['contrast_sensitivity_function' for i in range(21)]
    #magnitude is the gcf value
    df['magnitude'] = csf_df.loc[:,vid].values

    csf_events[vid] = df




