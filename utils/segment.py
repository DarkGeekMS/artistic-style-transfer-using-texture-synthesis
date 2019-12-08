import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.exposure import equalize_hist
from scipy.ndimage import convolve
import math
import copy

#for edge detection
def gradient(image):

    hx = np.array([[1, 2, 1], [0, 0, 0],  [-1, -2, -1]] ) /4.0
    hy = hx.T
    Ix = convolve(image, hx)
    Iy = convolve(image, hy)
    return np.sqrt(Ix**2+Iy**2)


def SobelRGB(image):
    r = image[:,:,0].astype('float32')
    g = image[:,:,1].astype('float32')
    b = image[:,:,2].astype('float32')
    X,Y = r.shape
    r /= 255
    g /= 255
    b /= 255
    r = gradient(r)
    b = gradient(b)
    g = gradient(g)
    Gm = np.zeros([X,Y])
    for i in range (0,X):
        Gm[i] = np.maximum( np.maximum(r[i] , g[i]), b[i])
    return Gm

def blurred(image):
    image = copy.deepcopy(image)
    conv = SobelRGB(image)
    return gaussian (conv , 20)



#even p -> object darker
#odd p -> object lighter

def morphChaneVese(image, itr=70 , smoothing= 2 , p=4):
    image = copy.deepcopy(image)
    image = rgb2gray(image)
    image = equalize_hist(image)
    s = image.shape
    grid = np.mgrid[[slice(i) for i in s]]
    grid = grid // p
    grid = grid & 1

    b = np.bitwise_xor.reduce(grid, axis=0)
    res = np.int8(b)

    u = np.int8(res > 0)
    for _ in range(itr):

        # inside = u > 0
        # outside = u <= 0
        c0 = (image * (1 - u)).sum() / float((1 - u).sum() + 1e-8)
        c1 = (image * u).sum() / float(u.sum() + 1e-8)

        # Image attachment
        du = np.gradient(u)
        abs_du = np.abs(du).sum(0)
        aux = abs_du * ((image - c1)**2 -  (image - c0)**2)

        u[aux < 0] = 1
        u[aux > 0] = 0

        # Smoothing
        for i in range(smoothing):
            u = gaussian (u , 1)
    return u
