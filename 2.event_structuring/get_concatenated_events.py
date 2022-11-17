# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 20:10:39 2021

@author: Hannah Craddock 
"""

"""
Created on Wed Jan 13 13:58:39 2021

@author: Hannah Craddock
"""

import pickle
import pandas as pd


#Data
with open('./events_per_movie.pickle','rb') as f:
    dict_all_events = pickle.load(f)
    
def get_df_concatenated_events(dict_all_events):
    'Concatentate events of all videos'
    
    #All videos
    dict_events_copy = dict_all_events.copy()
    df_events = pd.DataFrame()
       
    for vid, df_event in dict_events_copy.items():
        
        df_events = pd.concat([df_events, df_event])
        
    
    return df_events 

#Apply
df_events = get_df_concatenated_events(dict_all_events)

#Inspect
df_events['trial_type'].value_counts()

#Group-by 
df_events.loc[(df_events['trial_type'] == 'outside') or (df_events['trial_type'] == 'nature') or (df_events['trial_type'] == 'open'), ['trial_type', 'duration']].groupby('trial_type')['duration'].sum()

df_events[(df_events['trial_type'] == 'outside') | (df_events['trial_type'] == 'nature') | (df_events['trial_type'] == 'open')].groupby('trial_type')['magnitude'].sum()

df_events['trial_type'].value_counts()
