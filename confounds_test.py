import os
import pandas as pd
import skvideo.io
import numpy as np
from skimage import color

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
    outPath: path to save trimmed video, include extenstion '*.mp4'
    """
    ffmpegCommand = "ffmpeg -i " + inPath + " -ss  " + start + " -to " + end + " -c copy " + outPath
    print(ffmpegCommand)
    runBash(ffmpegCommand)

if __name__ == "__main__":
    
    vidPath = './test_stim'
    movie_times = pd.read_csv('./test_movie.csv',sep=';', index_col='title')

    for vid in os.listdir('vidPath'):
        #crop the video using ffmpeg command line
        start = movie_times.loc[vid,'start']
        end = movie_times.loc[vid,'end']
        inPath = os.path.join(vidPath,vid)
        outPath = f'{vidPath}/trimmed/{vid}'

        crop(start,end,inPath,outPath)
        print(f'{vid} trimmed')

        #load the cropped video
        metadata, singlevideo, dur, fps= load_video(outPath)
        print(f'{vid} loaded')

        # Convert to LAB
        lab_vid = np.zeros(singlevideo.shape)
        for ind, frame in enumerate(singlevideo):
            lab_vid[ind,:,:,:] = color.rgb2lab(frame)
