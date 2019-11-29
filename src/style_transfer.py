import numpy as np
from scipy.misc import imresize
from skimage.util import view_as_windows

from data_loader.loader import DataLoader
from ml_models import irls, pca, nn
from utils import color_transfer, denoise

def style_transfer(content_path, style_path, img_size, num_res, patch_sizes, sub_gaps, irls_iter, alg_iter, robust_stat):
    """ 
    The function to perform the main logic of style transfer,
    The main stylization loop. 
  
    Parameters: 
    content_path (string): Path to the content image (or video)
    style_path (string): Path to the style image
    img_size (int): Maximum image size (assuming square grid)
    num_res (int): Number of resolution layers for subsampling
    patch_sizes (list[int]): A list of all patch sizes to work with
    irls_iter (int): Number of IRLS algorithm iterations
  
    Returns: 
    ndarray: stylized image
  
    """
    # data loading ...
    data_gen = DataLoader(img_size)
    data_gen.prepare_data(content_path, style_path)

    # initialization ...
    # call color tranfer algorithm on content image
    # TODO

    # build gaussian pyramid
    content_layers = []
    style_layers = []
    seg_layers = []
    for iter in range(num_res-1, 0, -1):
        content_layers.append(imresize(data_gen.content, size=1/(2**iter), interp="bicubic"))
        style_layers.append(imresize(data_gen.style, size=1/(2**iter), interp="bicubic"))
        #seg_layers.append(imresize(data_gen.seg_mask, size=1/(2**iter), interp="bicubic"))  
    content_layers.append(data_gen.content)
    style_layers.append(data_gen.style)
    #seg_layers.append(data_gen.seg_mask)

    # setup patches
    style_patches = []
    for layer in style_layers:
        layer_patches = []
        for i in range(len(patch_sizes)):
            layer_patches.append(view_as_windows(layer, (patch_sizes[i],patch_sizes[i],3), sub_gaps[i]))
        style_patches.append(layer_patches) 
    
    # initialize X
    X = content_layers[0] + np.random.normal(0, 50, content_layers[0].shape)
    
    # main stylization loop ...
    for scale in range(num_res):
        for psize in patch_sizes:
            for iter in range(alg_iter):
                # patch matching
                # TODO

                # robust aggregation
                # TODO

                # content fusion
                # TODO

                # color transfer
                # TODO

                # denoise
                # TODO
                pass
                
        X = imresize(X, size=2.0, interp="bicubic")        