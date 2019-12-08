import numpy as np
import matplotlib.pyplot as plt

def show_images(images,titles=None):
    """
    Show the figures / plots inside the notebook

    Parameters:
    images (list[ndarray]): list of images to be displayed.
    titles (list[string]): list of titles of images (OPTIONAL).
    """
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()
