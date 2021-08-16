import os
import pandas as pd
import numpy as np
import pickle 

#inputs_path = './inputs_final'
inputs_path = '.\\inputs_final'

#turn framewise global contrast function results into an events dataframe, one per movie
gcf_df = pd.read_csv(os.path.join(inputs_path,'framewise_gcf_longlist_new.csv'), index_col=0)
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

#do the same for temporal contrast sensitivity function
csf_df = pd.read_csv(f'{inputs_path}/total_energy_videos_w_new_movies.csv')
csf_events = {k:None for k in csf_df.columns}

for idx, vid in enumerate(csf_df.columns):
    df = pd.DataFrame(columns=['onset','duration','trial_type','magnitude'])
    
    df['onset'] = list(range(21))
    df['duration'] = [1 for i in range(21)]
    df['trial_type'] = ['contrast_sensitivity_function' for i in range(21)]
    df['magnitude'] = csf_df.loc[:,vid].values

    csf_events[vid] = df

#root mean squared differences between frames
rms_df = pd.read_csv(f'{inputs_path}/framewise_rms_longlist_new.csv', index_col=0)
rms_events = {k:None for k in gcf_df.index}

for idx, vid in enumerate(rms_df.index):
    df = pd.DataFrame(columns=['onset','duration','trial_type','magnitude'])
    df['onset'] = np.arange(0, (1/25)*562, 1/25)
    df['duration'] = [1/25 for i in range(562)]
    df['trial_type'] = ['rms_difference' for i in range(562)]
    df['magnitude'] = rms_df.loc[vid].values

    rms_events[vid] = df

#get events files in same structure for semantic tags

def elan_events(path, n_raters=2):
    df_dict = {}
    for vid in os.listdir(path):
        if '.txt' in vid:
            elan_df = pd.read_csv(os.path.join(path,vid), sep='\t', header=None)
            elan_df.columns = ['tag',' ','start','stop','duration',' ']
            elan_df['tag'] = [tag.lower() for tag in elan_df['tag'].to_list()]

            df = pd.DataFrame(columns=['onset','duration','trial_type','magnitude'])
            df['onset'] = elan_df.loc[:,'start'].values
            df['duration'] = elan_df.loc[:,'duration'].values
            df['trial_type'] = elan_df.loc[:,'tag'].values
            #set magnitude=1 if magnitude missing as per the behaviour of nilearn
            df['magnitude'] = [1/n_raters for i in range(len(df))]

            df_dict[vid.replace('.txt','.mp4')] = df

    return df_dict

elan_emily_gk = elan_events(f'{inputs_path}/elan_emily_gk')
elan_emma_ad = elan_events(f'{inputs_path}/elan_emma_ad')

#use indices of a loaded dataframe to get list of all files
vid_list = gcf_df.index
#new csf file is missing piper
#vid_list = vid_list.drop('piper.mp4')

#get all events for each movie
all_events = {k:None for k in vid_list}
for vid in vid_list:
    if 'piper' not in vid:
        vid_events = pd.concat([gcf_events[vid], csf_events[vid], rms_events[vid], elan_emily_gk[vid], elan_emma_ad[vid]])
    else:
        vid_events = pd.concat([gcf_events[vid], rms_events[vid], elan_emily_gk[vid], elan_emma_ad[vid]])
    all_events[vid] = vid_events
#fill in piper from old events files
#with open('./events_per_movie_all.pickle','rb') as f:
#    all = pickle.load(f,encoding='latin1')
#all_events['piper.mp4'] = all['piper.mp4']

with open('./events_per_movie_longlist_new.pickle','wb') as f:
    pickle.dump(all_events,f)
