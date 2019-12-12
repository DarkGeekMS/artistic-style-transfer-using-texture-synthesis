import numpy as np
import cv2
import skimage.io as io
import matplotlib.pyplot as plt
from scipy import stats


# color transfer with histogram matching
def HM_color_transfer(content, style):
    """
        Color transfer the content image to the style image using cumulative
        distribution of both images,
        Args:
            content: content image in RGB space.
            style: target style in RGB space.

        Returns:
            Color transfer image in RGB space.
    """
    #copy style image and content image then convert them from 0:1 to 0:255 scale.
    transfered = np.copy(content)
    style = np.copy(style)
    transfered *=255
    style *= 255

    #calculate normalized cumulative histogram then update the content image based on the calculated values.
    for h in range (0,3):
        content_c = transfered[:,:,h]
        style_c   = style[:,:,h]
        height , width = content_c.shape
        contentValues,_,_,_ = stats.cumfreq(content_c, numbins=256)
        contentValues /= contentValues[-1]

        styleValues,_,_,_   = stats.cumfreq(style_c, numbins=256)
        styleValues /= styleValues[-1]

        K=256
        new_values=np.zeros((K))

        for a in np.arange(K):
            j=K-1
            while True:
                new_values[a]=j
                j=j-1
                if j<0 or contentValues[a]>styleValues[j]:
                    break

        for i in np.arange(height):
            for j in np.arange(width):
                a=content_c.item(i,j)
                b=new_values[int(a)]
                transfered[:,:,h].itemset((i,j),b)
        #transfered[:,:,h] = gaussian(transfered[:,:,h])
    #return the image to 0:1 scale
    transfered = transfered /255
    return transfered


def color_transfer(source, target):
    """
        Color transfer the source image to the target image using some
        stats extracted from both images in the LAB color space,
        Args:
            source: style image in RGB space.
            target: target image in RGB space.

        Returns:
            Color transfer image in RGB space.
        """
    # convert RGB color space of source and target image to LAB color space
    # note: OpenCV expects floats to be 32-bit, so use that instead of 64-bit
    source = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype("float32")
    target = cv2.cvtColor(target, cv2.COLOR_RGB2LAB).astype("float32")
    # compute color statistics for the source and target images
    (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source)
    (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target)

    # separate the color channels of the target
    # subtract the means from the target image
    (l, a, b) = cv2.split(target)
    l -= lMeanTar
    a -= aMeanTar
    b -= bMeanTar

    # scale by the standard deviations
    l = (lStdTar / lStdSrc) * l
    a = (aStdTar / aStdSrc) * a
    b = (bStdTar / bStdSrc) * b

    # add in the source mean
    l += lMeanSrc
    a += aMeanSrc
    b += bMeanSrc

    # clip the pixel intensities to [0, 255] if they fall outside this range
    l = np.clip(l, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)

    # merge the channels together and convert back to the RGB color
    # space, being sure to utilize the 8-bit unsigned integer data
    # type
    transfer = cv2.merge([l, a, b])
    transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2RGB)

    # return the color transferred image
    return transfer


def image_stats(image):
    """
    Extract mean and standard deviation of the given image
    for every color channel.

    Args:
    image: input image in LAB color space.
    Returns:
    mean and standard deviation for every color channel
    """
    # split color channels of the image
    (l, a, b) = cv2.split(image)
    # calculate mean and standard deviation for every color channel
    (lMean, lStd) = (l.mean(), l.std())
    (aMean, aStd) = (a.mean(), a.std())
    (bMean, bStd) = (b.mean(), b.std())

    # return the color statistics
    return lMean, lStd, aMean, aStd, bMean, bStd

