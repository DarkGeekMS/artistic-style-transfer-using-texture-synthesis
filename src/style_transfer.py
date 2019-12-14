import numpy as np
import cv2
from sklearn.feature_extraction.image import extract_patches
from skimage.util import view_as_windows, random_noise, pad
from sklearn.neighbors import NearestNeighbors

from data_loader.loader import DataLoader
from ml_models import irls, pca
from utils import color_transfer, denoise, utils

def style_transfer(content_path, style_path, img_size, num_res, patch_sizes, sub_gaps, irls_iter, alg_iter, robust_stat, content_weight, segmentation_mode, color_transfer_mode, denoise_sigma_s, denoise_sigma_r, denoise_iter):
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
    content_weight (float): Weight of content during fusion
    segmentation_mode (int): Edge Segmentation method to be used
    color_transfer_mode (int): Color Transfer method to be used
    denoise_sigma_s (float): sigma_s constant for denoise
    denoise_sigma_r (float): sigma_r constant for denoise
    denoise_iter (int): number of iterations for denoise

    Returns: 
    ndarray: stylized image
  
    """
    ### data loading ...
    print("Initializing Dataloader ...")
    data_gen = DataLoader(img_size)
    data_gen.prepare_data(content_path, style_path, segmentation_mode)

    ### initialization ...
    ## call color tranfer algorithm on content image
    print("Performing Color Transfer ...")
    if (color_transfer_mode==0):
        adapt_content = color_transfer.color_transfer(data_gen.style, data_gen.content)
    else:
        adapt_content = color_transfer.HM_color_transfer(data_gen.style, data_gen.content)
    
    ## build gaussian pyramid
    print("Building Pyramids ...")
    content_layers = []
    style_layers = []
    seg_layers = []
    content_layers.append(adapt_content)
    style_layers.append(data_gen.style)
    seg_layers.append(data_gen.seg_mask)
    for iter in range(num_res-1, 0, -1):
        content_layers.append(cv2.pyrDown(content_layers[-1].astype(np.float32)).astype(np.float32))
        style_layers.append(cv2.pyrDown(style_layers[-1].astype(np.float32)).astype(np.float32))
        seg_layers.append(cv2.pyrDown(seg_layers[-1].astype(np.float32)).astype(np.float32))    
    content_layers.reverse()      
    style_layers.reverse()  
    seg_layers.reverse()  

    ## initialize X
    print("Initializing Output ...")
    X = random_noise(content_layers[0], mode='gaussian', var=50)
    
    ### main stylization loop ...
    print()
    print("Starting Stylization ...")
    print()

    for s_index in range(num_res):
        print("Scale",s_index,": ")
        
        ## add some extra noise to the output
        X = random_noise(X, mode='gaussian', var=20 / 250.0)

        for p_index in range(len(patch_sizes)):
            print("Patch Size",patch_sizes[p_index],": ")

            ## pad content, style, segmentation mask and X for correct style mapping
            # calculate padding size value
            original_size = style_layers[s_index].shape[0]
            num_patches = int((original_size - patch_sizes[p_index]) / sub_gaps[p_index] + 1)	
            pad_size = patch_sizes[p_index] - (original_size  - num_patches * sub_gaps[p_index])
            pad_arr = ((0, pad_size), (0, pad_size), (0, 0))
            # pad all inputs
            current_style = pad(style_layers[s_index], pad_arr, mode='edge')
            current_seg = pad(seg_layers[s_index].reshape(seg_layers[s_index].shape[0], seg_layers[s_index].shape[1], 1), pad_arr, mode='edge')
            current_content = pad(content_layers[s_index], pad_arr, mode='edge')
            X = pad(X, pad_arr, mode='edge')
            
            ## extract style patches and fit nearest neighbors
            style_patches = extract_patches(current_style, patch_shape=(patch_sizes[p_index], patch_sizes[p_index], 3), extraction_step=sub_gaps[p_index])
            style_features = style_patches.reshape(-1, patch_sizes[p_index] * patch_sizes[p_index] * 3)
            proj_matrix = 0
            if (patch_sizes[p_index] <= 21):
                proj_matrix, proj_style_features = pca.pca(style_features)
                neighbors = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(proj_style_features)
            else:
                neighbors = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(style_features)

            for iter in range(alg_iter):
                print("Iteration",iter," ...")

                # patch matching
                X_patches = extract_patches(X, patch_shape=(patch_sizes[p_index], patch_sizes[p_index], 3), extraction_step=sub_gaps[p_index])

                # robust aggregation
                X = irls.IRLS(X, X_patches, style_patches, neighbors, proj_matrix, patch_sizes[p_index], sub_gaps[p_index], irls_iter, robust_stat)

                # content fusion
                X = (1.0 / (content_weight * current_seg + 1)) * (X + (content_weight * current_seg * current_content))

                # color transfer
                if (color_transfer_mode==0):
                    X = color_transfer.color_transfer(current_style, X)
                else:
                    X = color_transfer.HM_color_transfer(current_style, X)    

                # denoise
                X[:original_size, :original_size, :] = denoise.denoise_image(X[:original_size, :original_size, :], sigma_s=denoise_sigma_s, sigma_r=denoise_sigma_r, iterations=denoise_iter)

            X = X[:original_size, :original_size, :]  # back to the original size

        if (s_index != num_res-1):        
            X = cv2.resize(X.astype(np.float32), (content_layers[s_index+1].shape[0], content_layers[s_index+1].shape[1])).astype(np.float32)

        print()          

    print("Stylization Done!")
    # save and show stylized image
    im_to_write = cv2.cvtColor((X*255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite("outputs/output.png", im_to_write)
    utils.show_images([data_gen.content, data_gen.style, X], ["Content", "Style", "Stylized Image"])

