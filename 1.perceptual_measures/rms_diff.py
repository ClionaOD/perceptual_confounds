import math
from PIL import ImageChops

def rmsdiff(im1, im2):
    diff = ImageChops.difference(im1,im2)
    h = diff.histogram()
    sq = (value*((idx%256)**2) for idx, value in enumerate(h))
    sum_of_squares = sum(sq)
    rms = math.sqrt(sum_of_squares/float(im1.size[0] * im1.size[1]))
    return h,rms