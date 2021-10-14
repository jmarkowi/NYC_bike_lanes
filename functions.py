import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from PIL.ImageOps import exif_transpose
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pickle import dump

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

def visualize_results(results, model, train_gen, val_gen, pickle=True):
    '''
    Plot the training and validation data from a trained model, given the results/history.
    Plot accuracy, recall, precision, and loss.
    
    If model and generators are provided, print evaluation of training and validation data
    and plot confusion matricies.
    
    If pickle=True, pickle the training history (dictionary) for later analysis and 
    comparison.
    '''
    # Training history
    history = results.history
    
    # Pickle results
    if pickle:
        dump(history, open(f'{model}_history.pkl', 'wb'))
    
    # Plot metrics
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Training Results', fontsize=15)
    
    ax1 = axs[0][0]
    ax2 = axs[0][1]
    ax3 = axs[1][0]
    ax4 = axs[1][1]
    
    # Accuracy
    ax1.plot(history['val_acc'])
    ax1.plot(history['acc'])
    ax1.legend(['Validation acc', 'Training acc'])
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    
    # Loss
    ax2.plot(history['val_loss'])
    ax2.plot(history['loss'])
    ax2.legend(['Validation loss', 'Training loss'])
    ax2.set_title('Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    
    # Recall
    ax3.plot(history['val_recall'])
    ax3.plot(history['recall'])
    ax3.legend(['Validation recall', 'Training recall'])
    ax3.set_title('Recall')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Recall')

    # Precision
    ax4.plot(history['val_precision'])
    ax4.plot(history['precision'])
    ax4.legend(['Validation precision', 'Training precision'])
    ax4.set_title('Precision')
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('Precision')
    
    # Evaluations
    print('Training eval:')
    results_train = model.evaluate(train_gen)
    print('\nValidation eval:')
    results_val = model.evaluate(val_gen)
        
    # Confusion matrix for validation data
    y_val_preds = (model.predict(val_gen) > 0.5).astype('int32')
    ConfusionMatrixDisplay(confusion_matrix(val_gen.labels, y_val_preds),
                           display_labels=['Val_open_lane', 'Val_vehicle_lane']).plot();
        
    return results