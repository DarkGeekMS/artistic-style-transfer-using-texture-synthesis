import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import polygon
from scipy.ndimage import filters,convolve, binary_dilation
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.exposure import equalize_hist
from skimage.segmentation import (morphological_chan_vese,
                                  morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient,
                                  checkerboard_level_set)
import math
import copy


#for edge detection
def gradient(image):
    """
        Calculate sobel gradient magnitude for grayscaled image,
        Args:
            image: gray-scaled image.

        Returns:
            gray-scaled image of the magnitude of the gradient.
    """
    # vertical and horizontal filters decleration for gradient calculations.
    hx = np.array([[1, 2, 1], [0, 0, 0],  [-1, -2, -1]] ) /4.0
    hy = hx.T
    #convolve the image with the filters to detect vertical and horizontal lines
    Ix = convolve(image, hx)
    Iy = convolve(image, hy)
    #return the magnitude of each pixel gradient
    return np.sqrt(Ix**2+Iy**2)


def SobelRGB(image):

    """
        Calculate sobel gradient magnitude for each channel in RGB image,
        Args:
            image: RGB image.

        Returns:
            gray-scaled image of the magnitude of the gradient.
    """
    #change the image scale from 0:255 to 0:1 scale
    r = image[:,:,0].astype('float32')
    g = image[:,:,1].astype('float32')
    b = image[:,:,2].astype('float32')
    X,Y = r.shape
    r /= 255
    g /= 255
    b /= 255
    #calculate the gradient for each channel
    r = gradient(r)
    b = gradient(b)
    g = gradient(g)
    Gm = np.zeros([X,Y])
    #total image gradient is the maximum value of the three channels
    for i in range (0,X):
        Gm[i] = np.maximum( np.maximum(r[i] , g[i]), b[i])
    #return the final result
    return Gm

#segmentation mask by blurring
def segBlurring(image , sigma =20):
    """
        Segment RGB image using RGB-edge detection thenblurring the edges with high sigma (20),
        Args:
            image: RGB image.
            sigma: the sigma of the gaussian filter applied to the edged image, default = 20.

        Returns:
            gray-scaled image represent the segmentation mask.
    """
    image = copy.deepcopy(image)
    conv = SobelRGB(image)
    return gaussian (conv , sigma)



def cross(o, a, b):
    """
        Calculate the cross product between oa ans ob vectors,
        Args:
            o: RGB image.
            a:
            b:

        Returns:
            the cross product magnitude, positive value if oab makes CCW turn , negative for CW turn, zero if points are collinear.
    """
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def convex_hull(points):
    """
        Computes the convex hull of a set of 2D points.

        Args:
            points: list of 2D array representing the points.
        Returns:
            a list of vertices of the convex hull in counter-clockwise order,
            starting from the vertex with the lexicographically smallest coordinates.
    """

    # Sort the points.
    points.sort(key = lambda points: points[1])

    # Boring case: no points or a single point, possibly repeated multiple times.
    if len(points) <= 1:
        return points

    # Build lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Concatenation of the lower and upper hulls gives the convex hull.
    # Last point of each list is omitted because it is repeated at the beginning of the other list.
    return lower[:-1] + upper[:-1]

def fillObject(img):
    """
        fill the object detected using convex_hull algorithm and polygon drawing.

        Args:
            img: Binary image contains the edges of the object to be filled.
        Returns:
            Binary image represents the filled object.
    """

    image = img.copy()
    points = []
    #save the edge points to detect convex_hull over them
    for j in range(image.shape[1]):
        for i in range(image.shape[0]):
            if image[i, j] == 1:
                points.append([i, j])
    #calculate the convex_hull points
    l = convex_hull(points)
    l = np.array(l)
    r = l[:, 0]
    c = l[:, 1]
    #calculate the polygon that holds the object
    rr, cc = polygon(r, c)
    image[rr, cc] = 1
    #return the mask
    return image


def segMask(img , threshold= 0.5 ):
    """
        calculate the segmentation mask of the image by detecting the edges and filling it using convex_hull algorithm.
        Args:
            img: RGB image to calculate the segmentation mask over it.
            threshold: threshold to keep the strong edges of the image, default=0.5.
        Returns:
            Binary image represents the filled object.
    """
    img = img.copy()
    #calculate the edges using sobel_RGB edge detection
    img = SobelRGB(img)
    X,Y = img.shape
    #apply threshold to keep the strong edges.
    for i in range (0,X):
        for j in range (0,Y):
            if img[i][j] > threshold:
                img[i][j] = 1
            else:
                img[i][j] = 0

    # apply dilation to strength the edges.
    img = binary_dilation(img)

    #return convex_hull of the edged image
    img = fillObject(img)
    return img

def morphChaneVese(image, itr=120, p=3):
    """
        calculate the segmentation mask of the image using morphological_chan_vese built in function.
        Args:
            img: RGB image to calculate the segmentation mask over it.
            itr: number of iterations used to get the segmentation mask, default = 120.
            p: square width of the initial segmentation mask, default = 3.
        Returns:
            Binary image represents the filled object.
    """
    image = copy.deepcopy(image)
    #turn RGB image to gray scaled one.
    image = rgb2gray(image)
    #apply hestogram equalization to enhance the image.
    image = equalize_hist(image)
    #set the initial segmentation mask base on p value.
    init_ls = checkerboard_level_set(image.shape, p)
    #calculate the mask using morphological_chan_vese algorithm.
    mask = morphological_chan_vese(image, iterations=itr, init_level_set=init_ls, smoothing=2)
    #return the calculated mask
    return mask


def SegmentationMask(image , version = 2, mp=3, itrs =120, thresh =0.5, gsigma = 20):
    """
        calculate the segmentation mask of the image using morphological_chan_vese built in function.
        Args:
            image: RGB image to calculate the segmentation mask over it.
            version: represent the segmentation technique to be used , 0:blurred edges , 1:convex_hull , 2: morphological_chan_vese, default = 2.
            itrs: number of iterations used to get the segmentation mask for morphological_chan_vese, default = 120.
            pp: square width of the initial segmentation mask for morphological_chan_vese, default = 3.
            thresh: threshold the edges to keep strong edges only in convex_hull segmentation technique, default = 0.5.
            gsigma: sigma value used in gaussian blurring in blurred edges segmentation technique, default = 20.
        Returns:
            Binary image represents the filled object.
    """
    if version == 0:
        return segBlurring(image, gsigma)
    if version == 1:
        return segMask(image , thresh)
    #default version is morphological_chan_vese technique.
    return morphChaneVese(image, itrs, mp)