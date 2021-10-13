import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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