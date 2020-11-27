import os
import math
import numpy as np
import pandas as pd
from PIL import ImageChops
from perceptual_confounds import load_video

def rmsdiff(im1, im2):
    diff = ImageChops.difference(im1,im2)
    h = diff.histogram()
    sq = (value*((idx%256)**2) for idx, value in enumerate(h))
    sum_of_squares = sum(sq)
    rms = math.sqrt(sum_of_squares/float(im1.size[0] * im1.size[1]))
    return h,rms

vidPath = '/home/clionaodoherty/foundcog_stimuli/fps'
framewise_rms = {k:[] for k in os.listdir(vidPath)}
for vid in os.listdir(vidPath):
    metadata, singlevideo, dur, fps = load_video(f'{vidPath}/{vid}')
    print(f'{vid} loaded')
    
    all_rms = []
    for idx in range(singlevideo.shape[0] -1):
        h, rms = rmsdiff(singlevideo[idx,:,:,:], singlevideo[idx+1,:,:,:])
        all_rms.append(rmsdiff)
    
    framewise_rms[vid] = all_rms

rms_df = pd.DataFrame.from_dict(framewise_rms, orient='index')
rms_df.to_csv('./framewise_rms.csv')