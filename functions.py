import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from PIL.ImageOps import exif_transpose

def get_images(filepath, num_images=None):
    '''
    Given a filepath for a folder of images, return a list of those images
    as arrays of dtype=uint8.
    '''
    # List of filenames
    filenames = os.listdir(filepath)
    # List of full filepaths to each image
    filepaths = [os.path.join(filepath, name) for name in filenames]
    # Return list of files as raw image arrays
    if num_images:
        return [mpimg.imread(img) for img in filepaths[:num_images]]
    else:
        return [mpimg.imread(img) for img in filepaths]
    
def reorient_images(input_dir, output_dir):
    '''
    Given a filepath for a folder of images, iterate over all images
    in the folder and re-orient them by adjusting according to 
    'Orientation' EXIF tag.
    Save new images in place.
    '''
    # List of filenames
    filenames = os.listdir(input_dir)

    # Iterate over each image and re-orient, then save to output_dir
    for name in filenames:
        filepath = os.path.join(input_dir, name)
        img = Image.open(filepath)
        img = exif_transpose(img)
        img.save(os.path.join(output_dir, name))
    
    return
        