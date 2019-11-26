from data_loader.loader import DataLoader
from ml_models import irls, pca, nn
from utils import color_transfer, denoise

def style_transfer(content_path, style_path, img_size, num_res, patch_sizes, irls_iter):
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
    data_gen = DataLoader(img_size)
    datagen.prepare_data(content_path, style_path)
    # call color tranfer algorithm on content image
    # build gaussian pyramid
    # setup patches
    # main stylization loop