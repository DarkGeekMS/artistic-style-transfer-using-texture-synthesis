import numpy as np
import cv2
from sklearn.feature_extraction.image import extract_patches
from skimage.util import view_as_windows, random_noise
from sklearn.neighbors import NearestNeighbors

from data_loader.loader import DataLoader
from ml_models import irls, pca
from utils import color_transfer, denoise, utils

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
    print("Initializing Dataloader ...")
    data_gen = DataLoader(img_size)
    data_gen.prepare_data(content_path, style_path)

    # initialization ...
    # call color tranfer algorithm on content image
    print("Performing Color Transfer ...")
    data_gen.content = color_transfer.color_transfer(data_gen.style, data_gen.content)
    
    # build gaussian pyramid
    print("Building Pyramids ...")
    content_layers = []
    style_layers = []
    seg_layers = []
    content_layers.append(data_gen.content)
    style_layers.append(data_gen.style)
    seg_layers.append(data_gen.seg_mask)
    for iter in range(num_res-1, 0, -1):
        content_layers.append(cv2.pyrDown(content_layers[-1].astype(np.float32)).astype(np.float32))
        style_layers.append(cv2.pyrDown(style_layers[-1].astype(np.float32)).astype(np.float32))
        seg_layers.append(cv2.pyrDown(seg_layers[-1].astype(np.float32)).astype(np.float32))    
    content_layers.reverse()      
    style_layers.reverse()  
    seg_layers.reverse()  

    # setup patches
    print("Patching Style Layers ...")
    style_patches = []
    for layer in style_layers:
        layer_patches = []
        for i in range(len(patch_sizes)):
            layer_patches.append(extract_patches(layer, patch_shape=(patch_sizes[i],patch_sizes[i],3), extraction_step=sub_gaps[i]))
        style_patches.append(layer_patches) 
    
    # initialize X
    print("Initializing Output ...")
    X = random_noise(content_layers[0], mode='gaussian', var=20 / 250.0)
    
    # main stylization loop ...
    print()
    print("Starting Stylization ...")
    print()
    for s_index in range(num_res):
        print("Scale",s_index,": ")
        X = random_noise(X, mode='gaussian', var=20 / 250.0)
        for p_index in range(len(patch_sizes)):
            print("Patch Size",patch_sizes[p_index],": ")
            style_features = style_patches[s_index][p_index].reshape(-1, patch_sizes[p_index] * patch_sizes[p_index] * 3)
            proj_matrix, proj_style_features = pca.pca(style_features)
            neighbors = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(proj_style_features)
            for iter in range(alg_iter):
                print("Iteration",iter," ...")
                # patch matching
                X_patches = extract_patches(X, patch_shape=(patch_sizes[p_index], patch_sizes[p_index], 3), extraction_step=sub_gaps[p_index])

                # robust aggregation
                X = irls.IRLS(X, X_patches, style_patches[s_index][p_index], neighbors, proj_matrix, patch_sizes[p_index], sub_gaps[p_index], irls_iter, robust_stat)
                
                # content fusion
                seg_mask = seg_layers[s_index].reshape(X.shape[0], X.shape[1], 1)
                X = (1.0 / (seg_mask + 1)) * (X + (seg_mask * content_layers[s_index]))
                
                # color transfer
                X = color_transfer.color_transfer(style_layers[s_index], X)
                
                # denoise
                X = denoise.denoise_image(X)
                
        if (s_index != num_res-1):        
            X = cv2.resize(X.astype(np.float32), (content_layers[s_index+1].shape[0], content_layers[s_index+1].shape[1])).astype(np.float32)
        print()          

    print("Stylization Done!")
    utils.show_images([data_gen.content, data_gen.style, X], ["Content", "Style", "Stylized Image"])

