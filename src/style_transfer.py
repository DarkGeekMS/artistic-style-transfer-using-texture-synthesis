import numpy as np
from scipy.misc import imresize
from sklearn.feature_extraction.image import extract_patches
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
    for iter in range(num_res-1, 0, -1):
        content_layers.append(imresize(data_gen.content, size=1/(2**iter), interp="bicubic"))
        style_layers.append(imresize(data_gen.style, size=1/(2**iter), interp="bicubic"))
        seg_layers.append(imresize(data_gen.seg_mask, size=1/(2**iter), interp="bicubic"))  
    content_layers.append(data_gen.content)
    style_layers.append(data_gen.style)
    seg_layers.append(data_gen.seg_mask)

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
    X = content_layers[0] + np.random.normal(0, 50, content_layers[0].shape)
    
    # main stylization loop ...
    print()
    print("Starting Stylization ...")
    print()
    for s_index in range(num_res):
        print("Scale",s_index,": ")
        for p_index in range(len(patch_sizes)):
            print("Patch Size",patch_sizes[p_index],": ")
            for iter in range(alg_iter):
                print("Iteration",iter," ...")
                # patch matching
                style_features = style_patches[s_index][p_index].reshape(-1, patch_sizes[p_index] * patch_sizes[p_index] * 3)
                proj_matrix, proj_style_features = pca.pca(style_features)
                neighbors = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(proj_style_features)
                X_patches = extract_patches(X, patch_shape=(patch_sizes[p_index], patch_sizes[p_index], 3), extraction_step=sub_gaps[p_index])

                # robust aggregation
                X = irls.IRLS(X, X_patches, style_patches[s_index][p_index], neighbors, proj_matrix, patch_sizes[p_index], sub_gaps[p_index], irls_iter, robust_stat)
                
                # content fusion
                X = np.dot(np.linalg.inv(seg_layers[s_index] + np.diag(np.repeat(1, seg_layers[s_index].shape[0]))), (X + np.dot(seg_layers[s_index], content_layers[s_index])))
                
                # color transfer
                X = color_transfer.color_transfer(style_layers[s_index], X)
                
                # denoise
                X = denoise.denoise_image(X)
                
        if (s_index != num_res-1):        
            X = imresize(X, size=2.0, interp="bicubic").astype(np.float32)  
        print()          

    print("Stylization Done!")
    utils.show_images([data_gen.content, data_gen.style, X], ["Content", "Style", "Stylized Image"])

