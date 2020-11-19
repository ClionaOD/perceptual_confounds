import os
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

def crop(start,end,input,output):
    """
    args:
    start: start time for trimmed video in form hh:mm:ss
    end: end time for trimmed video in form hh:mm:ss
    input: path to video to trim, include extension '*.mp4'
    output: path to save trimmed video, include extenstion '*.mp4'
    """
    ffmpegCommand = "ffmpeg -i " + input + " -ss  " + start + " -to " + end + " -c copy " + output
    print(ffmpegCommand)
    runBash(ffmpegCommand)

if __name__ == "__main__":
    vidPath = ''

    for vid in os.listdir('vidPath'):

metadata, singlevideo, dur, fps= load_video(vid)

# Convert to LAB
lab_vid = np.zeros(singlevideo.shape)
for ind, frame in enumerate(singlevideo):
    lab_vid[ind,:,:,:] = color.rgb2lab(frame)
