import os 
import numpy as np
from scipy.misc import imread, imsave, imresize

from utils.segment import morphChaneVese

class DataLoader:
    """ 
    This is a class for data loading and storing. 
      
    Attributes: 
        img_size (int): Maximum image size. 
        content (ndarray): Content image grid.
        style (ndarray): Style image grid.
        seg_mask (ndarray): Segmentation mask grid.
        vid_ext (list[string]): Possible video extensions.
        img_ext (list[string]): Possible image extensions.
    """
    def __init__(self, img_size=400):
        """ 
        The constructor for DataLoader class. 
  
        Parameters: 
           img_size (int): Maximum image size (assuming square grid).   
        """
        self.img_size = img_size
        self.content = None
        self.style = None
        self.seg_mask = None
        self.vid_ext = ['.avi', '.mp4', '.mkv', '.wmv']
        self.img_ext = ['.tif','.png', '.jpg', 'jpeg']

    def load_content(self, file_path):
        """ 
        The function to load content image (or video). 
  
        Parameters: 
            file_path (string): Path to the content image (or video). 
        """
        if file_path[-4:] in self.img_ext:
            self.content = imresize(imread(file_path), tuple((self.img_size,self.img_size))) / 255.0
        elif file_path[-4:] in self.vid_ext:   
            pass # to be implemented in case of video stylization 
        else:
            raise Exception('Format incompatible!')

    def load_style(self, file_path):
        """ 
        The function to load style image. 
  
        Parameters: 
            file_path (string): Path to the style image. 
        """
        if file_path[-4:] in self.img_ext:
            self.style = imresize(imread(file_path), tuple((self.img_size,self.img_size))) / 255.0
        else:
            raise Exception('Format incompatible!')    

    def segment_content(self):
        """ 
        The function to call the segmentation algorithm on content image. 
        """
        self.seg_mask = morphChaneVese(self.content)

    def prepare_data(self, content_path, style_path):
        """ 
        The function to encapsulate all the data loader functionalities. 
  
        Parameters: 
            content_path (string): Path to the content image (or video). 
            style_path (string): Path to the style image. 
        """
        self.load_content(content_path)
        self.load_style(style_path)
        self.segment_content()

    def reset_loader(self):
        """ 
        The function to reset the data loader and free used memory.  
        """
        del self.content
        del self.style  
        del self.seg_mask
        self.content = None  
        self.style = None
        self.seg_mask = None    