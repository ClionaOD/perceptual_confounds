import os
import pandas as pd
import skvideo.io
import numpy as np
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
        fps: desired framerate
        inPath: path to video to change fps, include extension '*.mp4'
        outPath: path to save video, must be separate to inPath, include extension '*.mp4'
    """
    ffmpegCommand = f"ffmpeg -i {inPath} -filter:v fps=fps={fps} {outPath}"
    print(ffmpegCommand)
    runBash(ffmpegCommand)

def crop_movies(vid_path, movie_times):
    """
    crop all videos in vidPath according to start/end in movie_times
    
    args: 
        vid_path: the path to find videos to trim
        movie_times: pandas dataframe containing cols:[vid_title, start_time, end_time]
    """
    dur = '00:00:22.500'
    for vid in os.listdir(vidPath):
        if not os.path.isdir(f'{vidPath}/{vid}'):
            
            #crop the video using ffmpeg command line
            start = movie_times.loc[vid,'start']
            inPath = f'{vidPath}/{vid}'
            outPath = f'{vidPath}/trimmed/{vid}'

            crop(start,dur,inPath,outPath)
            print(f'{vid} trimmed')

            #set to mean frame rate fps=25
            change_framerate(inPath=f'{vidPath}/trimmed/{vid}', outPath=f'{vidPath}/fps/{vid}')
            print(f'{vid} frame rate standardised')

def get_gcf(vid, vidname, frame_dict, mean_dict):
    """
    args:
        vid: the video get contrast of
        frame_df: dictionary to store framewise gcf values with k=vid and v=list
        mean_df: dictionary to store mean gcf values with k=vid and v=[mean]
    """
    
    all_gcfs = np.zeros(vid.shape[0])
    for idx, frame in enumerate(vid):
        all_gcfs[idx] = compute_global_contrast_factor(frame)
        
    frame_dict[vidname] = all_gcfs
    mean = np.mean(all_gcfs)
    mean_dict[vidname] = [mean]

    print(f'{vid} mean gcf = {mean}')


def get_confounds(vidPath):
    #TODO: include rms
    """
    calculate gcf using code from Matkovic et al for each video
    args:  
        vidPath: path to the original videos, as in crop_movies
    returns:
        framewise_gcf: a pandas dataframe with indices=movie_titles and values=gcf for each frame
        mean_gcf: the mean gcf for each video
    """

    framewise_gcf = {k:[] for k in os.listdir(f'{vidPath}')}
    mean_gcf = {k:[] for k in os.listdir(f'{vidPath}')}
    
    #load the cropped video
    for vid in os.listdir(f'{vidPath}'):
        metadata, singlevideo, dur, fps = load_video(f'{vidPath}/{vid}')
        print(f'{vid} loaded')
        
        get_gcf(singlevideo, vid, framewise_gcf, mean_gcf)

    framewise_gcf = pd.DataFrame.from_dict(framewise_gcf, orient='index')
    mean_gcf = pd.DataFrame.from_dict(mean_gcf, orient='index')
    for vid in mean_gcf.index:
        mean_gcf.loc[vid,'std'] = framewise_gcf.loc[vid].std()
    mean_gcf.colums = ['mean','std']

    return framewise_gcf, mean_gcf

if __name__ == "__main__":
    
    vidPath = '/home/clionaodoherty/foundcog_stimuli'
    movie_times = pd.read_csv('./movie_times.csv',sep=';', index_col='title')
    
    load_video(f'{vidPath}/trimmed/minions_supermarket.mp4')
    load_video(f'{vidPath}/fps/minions_supermarket.mp4')
    """#crop all videos in vidPath according to start/end in movie_times
    #crop_movies(vidPath, movie_times)

    #calculate global contrast function and save the dataframes
    framewise_gcf, mean_gcf = get_confounds(f'{vidPath}/fps')
    framewise_gcf.to_csv('./framewise_gcf.csv')
    mean_gcf.to_csv('./mean_gcf.csv')

    df = pd.DataFrame(index=os.listdir(vidPath),columns=['fps'])
    for vid in os.listdir(vidPath):
        if not os.path.isdir(f'{vidPath}/{vid}'):
            metadata, singlevideo, dur, fps = load_video(f'{vidPath}/trimmed/{vid}')
            df.loc[vid]=fps
    df.to_csv('./fps_redo.csv')"""