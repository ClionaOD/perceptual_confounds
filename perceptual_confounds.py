import os
import pandas as pd
import skvideo.io
import numpy as np
import json
from skimage import color

from gcf import compute_global_contrast_factor, compute_image_average_contrast
#from rms_dff import rmsdiff

def load_video(infn, clipheight=None):
    # Load metadata
    metadata = skvideo.io.ffprobe(infn)
    dur=float(metadata['video']['@duration'])       
    nframes=int(float(metadata['video']['@nb_frames']))
    fps=nframes/dur
    print('Video %s duration %f nframes %d fps %f'%(infn,dur,nframes,fps))

    # Load a video
    singlevideo=skvideo.io.vread(infn)
    singlevideo=singlevideo.astype(np.uint8)

    # Clip height?
    if clipheight:
        h=singlevideo.shape[1]
        singlevideo=singlevideo[:,round(h/2-clipheight/2):round(h/2+clipheight/2),:,:]
        metadata['video']['@height']=clipheight
        print('Clipped height to %s'%clipheight)
    
    return metadata,singlevideo,dur,fps

def runBash(command):
    os.system(command)

def crop(start,dur,inPath,outPath):
    """
    args:
        start: start time for trimmed video in form hh:mm:ss
        dur: the length of desired clip in form hh:mm:ss. e.g. 23 s from start is 00:00:23
        inPath: path to video to trim, include extension '*.mp4'
        outPath: path to save trimmed video, include extension '*.mp4'
    """
    ffmpegCommand = f"ffmpeg -i {inPath} -ss {start} -t {dur} {outPath}"
    print(ffmpegCommand)
    runBash(ffmpegCommand)

def change_framerate(inPath,outPath,fps='25'):
    """
    args:
        inPath: path to video to change fps, include extension '*.mp4'
        outPath: path to save video, must be separate to inPath, include extension '*.mp4'
        fps: desired framerate
    """
    ffmpegCommand = f"ffmpeg -i {inPath} -filter:v fps=fps={fps} {outPath}"
    print(ffmpegCommand)
    runBash(ffmpegCommand)

def control_aspect(inPath, outPath, w=640, h=360):
    """
    args:
        inPath: path to video to change aspect ratio, include extension '*.mp4'
        outPath: path to save video, must be separate to inPath, include extension '*.mp4'
        w: desired width
        h: desired height
    """
    ffmpegCommand = f'ffmpeg -i {inPath} -vf scale={w}:{h} {outPath}'
    print(ffmpegCommand)
    runBash(ffmpegCommand)

def crop_movies(vid_path, movie_times):
    """
    crop all videos in vidPath according to start in movie_times for set duration of 22.5 sec
    
    args: 
        vid_path: the path to find videos to trim
        movie_times: pandas dataframe containing cols:[vid_title, start_time, end_time]
    """
    dur = '00:00:22.500'
    for vid in os.listdir(vidPath):
        if 'mp4' in f'{vidPath}/{vid}':
            
            if not vid in os.listdir(f'{vidPath}/trimmed'):
                #crop the video using ffmpeg command line
                start = movie_times.loc[vid,'start']
                inPath = f'{vidPath}/{vid}'
                outPath = f'{vidPath}/trimmed/{vid}'

                crop(start,dur,inPath,outPath)
                print(f'{vid} trimmed')

            if not vid in os.listdir(f'{vidPath}/fps'):
                #set to mean frame rate fps=25
                change_framerate(inPath=f'{vidPath}/trimmed/{vid}', outPath=f'{vidPath}/fps/{vid}')
                print(f'{vid} frame rate standardised')

            if not vid in os.listdir(f'{vidPath}/aspect'):
                #set to same aspect ratio
                control_aspect(inPath=f'{vidPath}/fps/{vid}', outPath=f'{vidPath}/aspect/{vid}')
                print(f'{vid} aspect ratio standardised')

def get_gcf(vid):
    """
    args:
        vid: the video get contrast of
        frame_df: dictionary to store framewise gcf values with k=vid and v=list
        mean_df: dictionary to store mean gcf values with k=vid and v=[mean]
    """
    
    all_gcfs = np.zeros(vid.shape[0])
    for idx, frame in enumerate(vid):
        all_gcfs[idx] = compute_global_contrast_factor(frame)
    
    return all_gcfs


def get_confounds(vidPath):
    """
    calculate gcf using code from Matkovic et al for each video
    args:  
        vidPath: path to the original videos, as in crop_movies
    returns:
        framewise_gcf: a pandas dataframe with indices=movie_titles and values=gcf for each frame
        mean_gcf: the mean gcf for each video
    """

    framewise_gcf = {k:[] for k in os.listdir(f'{vidPath}') if '.mp4' in k}
    mean_gcf = {k:[] for k in os.listdir(f'{vidPath}') if '.mp4' in k}
    
    #load the cropped video
    for vid in framewise_gcf.keys():
        metadata, singlevideo, dur, fps = load_video(f'{vidPath}/{vid}')
        print(f'{vid} loaded')
        
        all_gcfs = get_gcf(singlevideo)
        framewise_gcf[vid] = all_gcfs
        mean = np.mean(all_gcfs)
        mean_gcf[vid] = [mean]

        print(f'{vid} mean gcf = {mean}')

    framewise_gcf = pd.DataFrame.from_dict(framewise_gcf, orient='index')
    
    return framewise_gcf, mean_gcf
    """
    mean_gcf = pd.DataFrame.from_dict(mean_gcf, orient='index')
    mean_gcf.columns = ['mean','std']
    for vid in mean_gcf.index:
        mean_gcf.loc[vid,'std'] = framewise_gcf.loc[vid].std()

    return framewise_gcf, mean_gcf"""

if __name__ == "__main__":
    
    vidPath = '/home/clionaodoherty/foundcog_stimuli'
    movie_times = pd.read_csv('./movie_times.csv',sep=';', index_col='title')
    
    #crop all videos in vidPath according to start/end in movie_times
    #crop_movies(vidPath, movie_times)

    #calculate global contrast function and save the dataframes
    framewise_gcf, mean_gcf = get_confounds(f'{vidPath}/longlist')
    framewise_gcf.to_csv('./framewise_gcf_longlist_new.csv')
    #with open('./mean_gcf_longlist_new.json','w') as f:
    #    json.dump(mean_gcf, f)
    mean_gcf.to_csv('./mean_gcf_new.csv')