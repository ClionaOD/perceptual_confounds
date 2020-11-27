import math
from PIL import ImageChops

def rmsdiff(im1, im2):
    diff = ImageChops.difference(im1,im2)
    h = diff.histogram()
    sq = (value*((idx%256)**2) for idx, value in enumerate(h))
    sum_of_squares = sum(sq)
    rms = math.sqrt(sum_of_squares/float(im1.size[0] * im1.size[1]))
    return h,rms

vidPath = '/home/clionaodoherty/foundcog_stimuli'
framewise_gcf = {k:[] for k in os.listdir(f'{vidPath}/trimmed')}
for vid in os.listdir(f'{vidPath}/trimmed'):
    metadata, singlevideo, dur, fps = load_video(f'{vidPath}/trimmed/{vid}')
    print(f'{vid} loaded')
    
    all_gcfs = np.zeros(singlevideo.shape[0])
    for idx, frame in enumerate(singlevideo):
        all_gcfs[idx] = compute_global_contrast_factor(frame)
    
    framewise_gcf[vid] = all_gcfs