# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 14:01:06 2021

@author: Trish MacKeogh
"""
import seaborn as sn
import pickle
import pandas as pd
import numpy as np
import random
from nilearn.glm.first_level import make_first_level_design_matrix
#from nilearn.plotting import plot_design_matrix
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)

#Plot properties
%matplotlib qt

SMALL_SIZE = 8
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#Data
with open('./events_per_movie.pickle','rb') as f:
    all_events = pickle.load(f)
    
#Params
tr = 1.0
n_scans = 22
frame_times = np.arange(n_scans) * tr

def efficiency_calc(X, contrasts):
    '''Calculate efficiency for a given design matrix (i.e a given video) '''     
    
    #Singular matrix - no solution to inverse 
    print(X.shape)
    print(X.head(20))
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
    '''Get concatenated events dataframe. A period of rest in between each video is included'''
    
    #List of videos + randomise
    list_videos = list(all_events.keys())
    random.shuffle(list_videos) #Randomise movie order
    
    #Params
    movie_length = 22.524
    delay = 0.001
    df_rest = get_rest_df(rest_length)
    df_events_concat = pd.DataFrame()
    
    for idx, vid in enumerate(list_videos):      
        df_eventX = all_events[vid]
        df_temp = pd.concat([df_eventX, df_rest])
        #Adjust onsets of event
        df_temp['onset'] = df_temp['onset'] + idx*(movie_length + rest_length + delay)
        
        #Concatenate
        df_events_concat = pd.concat([df_events_concat,  df_temp])
    
    return df_events_concat

 
def get_efficiencies_LVO(all_events_dict, required_video_num, list_contrast_catgs):
    
    '''Using a leave one out principle to determine most 'efficient' videos.
    Remove one video at a time and calculate the updated efficiency.
    Repeat for each video. Drop the video that corresponds to the greatest efficiency when it is left out in the loo iteration.
    Repeat until required number of videos remains'''
    
    #Repeat until required_video_num of videos remains 
    while(len(all_events_dict) > required_video_num):        

        #Dict to store efficiencies
        dict_lvo_efficiencies = {vid: None for vid in all_events_dict.keys()}
               
        for vid in all_events_dict.keys():
            
            #Drop one video
            lvo_dict = {key: value for key, value in all_events_dict.items() if key != vid}
            
            #Get updated efficiencies - when leave one out 
            df_events = get_df_events(lvo_dict)
            print(df_events.tail(30))
            X = make_first_level_design_matrix(frame_times, df_events, hrf_model='glover')
            #Get contrasts
            contrast_vec = create_contrast_vec(list_contrast_catgs, X)
            #Calculate efficiency 
            dict_lvo_efficiencies[vid] = efficiency_calc(X, contrast_vec)
            
        #Find video which corresponds to the highest efficiency when dropped
        vid_max_improvement = max(dict_lvo_efficiencies, key=dict_lvo_efficiencies.get)
        #Drop from total events
        all_events_dict.pop(vid_max_improvement)
    
    return all_events_dict

def create_contrast_vec(list_contrast_catgs, X):
    'Create vector of contrasts'
    
    contrast_vec = np.zeros(X.shape[1]) #Set all to zero - default value unless categories present
    
    #Positive contrast (for example animate)
    if list_contrast_catgs[0] in X.columns:

        idx_pos = X.columns.get_loc(list_contrast_catgs[0])
        contrast_vec[idx_pos] = 1 #Set contrast to 1 at position in vector corresponding to catgory in design matrix
        
        #Set remaining conditions to be negative contrast
        for neg_catg in list_contrast_catgs[1:]:
            if neg_catg in X.columns:
                idx_pos = X.columns.get_loc(neg_catg) #Get column index of category
                contrast_vec[idx_pos] = -1
    
    #print(contrast_vec)
    return contrast_vec

#Apply functions
required_video_num = 8
list_contrast_catgs = ['animate', 'inanimate_small', 'inanimate_big']
events_top_movies = get_efficiencies_LVO(all_events, required_video_num, list_contrast_catgs)
 
#**********************************************************************************************
#Correlation

def get_df_all_videos(all_events):
    '''Get concatenated events dataframe. A period of rest in between each video is included'''
    
    #List of videos + randomise
    list_videos = list(all_events.keys())
    random.shuffle(list_videos) #Randomise movie order
    
    #Params
    movie_length = 22.524
    delay = 0.001
    df_all_videos = pd.DataFrame()
    
    for idx, vid in enumerate(list_videos):      
        df_videoX = all_events[vid]
        #Adjust onsets of event
        df_videoX['onset'] = df_videoX['onset'] + idx*(movie_length + delay)
     
        #Concatenate
        df_all_videos = pd.concat([df_all_videos,  df_videoX])
    
    return df_all_videos
 
def get_correlation_matrix(df_all_videos):
    
    #Design matrix
    
    X = make_first_level_design_matrix(frame_times, df_all_videos, hrf_model='glover') 
    
    #Correlation matrix
    corrMatrix = X.corr()
    sns.heatmap(corrMatrix, annot = True)
    plt.show()
    
    return X 
    
def repeat_corr(n_repeats)

    #Get df all events
    df_all_videos = get_df_all_videos(all_events)      
    #Corr matrix
    corrMatrix =  get_correlation_matrix(df_all_videos) 


#Get df all events
df_all_videos = get_df_all_videos(all_events)      
#Corr matrix
X = get_correlation_matrix(df_all_videos) 
corrMatrix =  get_correlation_matrix(df_all_videos)    