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
    print(singlevideo.shape)
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

def crop(start,end,inPath,outPath):
    """
    args:
    start: start time for trimmed video in form hh:mm:ss
    end: end time for trimmed video in form hh:mm:ss
    inPath: path to video to trim, include extension '*.mp4'
    outPath: path to save trimmed video, include extension '*.mp4'
    """
    ffmpegCommand = "ffmpeg -i " + inPath + " -ss  " + start + " -to " + end + " -c copy " + outPath
    print(ffmpegCommand)
    runBash(ffmpegCommand)

def crop_movies(vid_path, movie_times):
    """
    crop all videos in vidPath according to start/end in movie_times
    
    args: 
        vid_path: the path to find videos to trim
        movie_times: pandas dataframe containing cols:[vid_title, start_time, end_time]
    """
    for vid in os.listdir(vidPath):
        if not os.path.isdir(f'{vidPath}/{vid}'):
            #crop the video using ffmpeg command line
            start = movie_times.loc[vid,'start']
            end = movie_times.loc[vid,'end']
            inPath = os.path.join(vidPath,vid)
            outPath = f'{vidPath}/trimmed/{vid}'

            crop(start,end,inPath,outPath)
            print(f'{vid} trimmed')

def get_gcf(vidPath):
    #TODO: make gcf a separate function
    #TODO: make this a get_confounds function including rms
    """
    calculate gcf using code from Matkovic et al for each video
    args:  
        vidPath: path to the original videos, as in crop_movies
    returns:
        framewise_gcf: a pandas dataframe with indices=movie_titles and values=gcf for each frame
        mean_gcf: the mean gcf for each video
    """
    
    framewise_gcf = {k:[] for k in os.listdir(f'{vidPath}/trimmed')}
    mean_gcf = {k:[] for k in os.listdir(f'{vidPath}/trimmed')}
    
    #load the cropped video
    for vid in os.listdir(f'{vidPath}/trimmed'):
        metadata, singlevideo, dur, fps = load_video(f'{vidPath}/trimmed/{vid}')
        print(f'{vid} loaded')
        
        all_gcfs = np.zeros(singlevideo.shape[0])
        for idx, frame in enumerate(singlevideo):
            all_gcfs[idx] = compute_global_contrast_factor(frame)
        
        framewise_gcf[vid] = all_gcfs
        mean = np.mean(all_gcfs)
        mean_gcf[vid] = [mean]

        print(f'{vid} mean gcf = {mean}')

    framewise_gcf = pd.DataFrame.from_dict(framewise_gcf, orient='index')
    mean_gcf = pd.DataFrame.from_dict(mean_gcf, orient='index')
    for vid in mean_gcf.index:
        mean_gcf.loc[vid,'std'] = framewise_gcf.loc[vid].std()
    mean_gcf.colums = ['mean','std']

    return framewise_gcf, mean_gcf

if __name__ == "__main__":
    
    vidPath = '/home/clionaodoherty/foundcog_stimuli'
    """movie_times = pd.read_csv('./movie_times.csv',sep=';', index_col='title')
    
    #crop all videos in vidPath according to start/end in movie_times
    #TODO:make movie crops same length (22.5 s)
    #TODO:make movie crops same frame rate
    crop_movies(vidPath, movie_times)

    #calculate global contrast function and save the dataframes
    framewise_gcf, mean_gcf = get_gcf(vidPath)
    framewise_gcf.to_csv('./framewise_gcf.csv')
    mean_gcf.to_csv('./mean_gcf.csv')"""

    df = pd.DataFrame(index=os.listdir(vidPath),columns=['fps'])
    for vid in os.listdir(vidPath):
        if not os.path.isdir(f'{vidPath}/{vid}'):
            metadata, singlevideo, dur, fps = load_video(f'{vidPath}/trimmed/{vid}')
            df.loc[vid]=fps
    df.to_csv('./fps.csv')