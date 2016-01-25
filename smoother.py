import scipy.ndimage
import numpy as np

def gaussian_filter(x):
    s = scipy.ndimage.gaussian_filter(np.array(x, dtype=float), 100)
    return s