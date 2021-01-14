# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 14:01:06 2021

@author: Trish MacKeogh
"""
import pickle
import pandas as pd
import numpy as np
import random
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix
import matplotlib.pyplot as plt

#Functions

def efficiency_calc(X, contrast_vec):
    '''Calculate efficiency for a given design matrix (i.e a given video) '''       
    invXtX = np.linalg.inv(X.T.dot(X))
    efficiency = np.trace((1.0/ contrasts.T.dot(invXtX).dot(contrasts)))
    
    return efficiency

def get_rest_df(rest_length):
    
    df_rest = pd.DataFrame()
    df_rest['onset'] = 22.525
    df_rest['duration'] = rest_length
    df_rest['trial_type'] = 'rest'
    df_rest['magnitude'] = 1.0
    
    return df_rest

def get_df_events(all_events, rest_length = 2.0):
    ''' '''
    #List of videos + randomise
    list_videos = all_events.keys()
    random.shuffle(list_videos) #Randomise movie order
    
    #Params
    movie_length = 22.524
    delay = 0.01
    df_rest = get_rest_df(rest_length)
    df_events_concat = pd.DataFrame()
    
    for idx, vid in enumerate(list_videos):
        
        df_eventX = all_events[vid]
        df_temp = pd.concat([df_eventX, df_rest])
        #Adjust onsets of event
        df_temp['onset'] = df_event['onset'] + idx*(movie_length + rest_length + delay)
        
        #Concatenate
        df_events_concat = pd.concat([df_events_concat,  df_temp])

 
def get_efficiencies_LVO(all_events_dict, required_video_num, contrast_vec):
    
    #Repeat until required_video_num of videos remains 
    while(all_events_dict < required_video_num):        

        #Dict to store efficiencies
        dict_lvo_efficiencies = {vid: None, for vid in all_events_dict.keys()}
               
        for vid in all_events_dict.keys():
            
            #Drop one video
            lvo_dict = {key: value for key, value in all_events_dict.items() if key != vid}
            
            #Get updated efficiencies - when leave one out 
            df_events = get_df_events(lvo_dict)
            X = make_first_level_design_matrix(frame_times, df_events, hrf_model='glover')
            #Calculate efficiency 
            dict_lvo_efficiencies[vid] = efficiency_calc(X, contrast_vec)
            
        #Find video which corresponds to the highest efficiency when dropped
        vid_max_improvement = max(dict_test, key=dict_test.get)
        #Drop from total events
        all_events_dict.pop(vid_max_improvement)
    
    return all_events_dict

#Apply functions
required_video_num = 8
contrasts_vec = 
events_top_movies = get_efficiencies_LVO(all_events_dict, required_video_num, contrast_vec)
        
        
#Inspect all events
#Find where animate is & inanimate is 

def create_contrast_vec(list_contrasts, X):
    'Create vector of contrasts. If value not present?'
    
    idx_contrasts = []
    
    for contrast in list_contrasts:
        if contrast in df.columns:
    
    for idx, col in enumerate(df.columns):
        if col in list_contrasts:
            idx_contrasts.append(idx)
            
            
        
    

        
    
    