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

vid = './bathsong.mp4'
metadata, singlevideo, dur, fps= load_video(vid)

lab_vid = []
for frame in singlevideo:
    lab_img = color.rgb2lab(frame)
    lab_vid.append(lab_img)
print(len(lab_vid))
print(type(lab_vid[0]))

lab_arr = np.array(lab_vid)
print(lab_arr.shape)