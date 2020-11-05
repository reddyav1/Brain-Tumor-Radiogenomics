# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 19:43:28 2020

@author: reddyav1
"""

from glob import glob
import os

import numpy as np
from skimage.exposure import rescale_intensity
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage import img_as_ubyte

import argparse

def crop_sample(x,thresh):
    volume, mask = x
    # Zero out pixels below a threshold
    volume[volume < np.max(volume) * thresh] = 0
    z_projection = np.max(np.max(np.max(volume, axis=-1), axis=-1), axis=-1)
    z_nonzero = np.nonzero(z_projection)
    z_min = np.min(z_nonzero)
    z_max = np.max(z_nonzero) + 1
    y_projection = np.max(np.max(np.max(volume, axis=0), axis=-1), axis=-1)
    y_nonzero = np.nonzero(y_projection)
    y_min = np.min(y_nonzero)
    y_max = np.max(y_nonzero) + 1
    x_projection = np.max(np.max(np.max(volume, axis=0), axis=0), axis=-1)
    x_nonzero = np.nonzero(x_projection)
    x_min = np.min(x_nonzero)
    x_max = np.max(x_nonzero) + 1
    return (
        volume[z_min:z_max, y_min:y_max, x_min:x_max],
        mask[z_min:z_max, y_min:y_max, x_min:x_max],
    )


def pad_sample(x):
    volume, mask = x
    a = volume.shape[1]
    b = volume.shape[2]
    if a == b:
        return volume, mask
    diff = (max(a, b) - min(a, b)) / 2.0
    if a > b:
        padding = ((0, 0), (0, 0), (int(np.floor(diff)), int(np.ceil(diff))))
    else:
        padding = ((0, 0), (int(np.floor(diff)), int(np.ceil(diff))), (0, 0))
    mask = np.pad(mask, padding, mode="constant", constant_values=0)
    padding = padding + ((0, 0),)
    volume = np.pad(volume, padding, mode="constant", constant_values=0)
    return volume, mask


def resize_sample(x, size=256):
    volume, mask = x
    v_shape = volume.shape
    out_shape = (v_shape[0], size, size)
    mask = resize(
        mask,
        output_shape=out_shape,
        order=0,
        mode="constant",
        cval=0,
        anti_aliasing=False,
    )
    out_shape = out_shape + (v_shape[3],)
    volume = resize(
        volume,
        output_shape=out_shape,
        order=2,
        mode="constant",
        cval=0,
        anti_aliasing=False,
    )
    return volume, mask

def normalize_volume(volume):
    p10 = np.percentile(volume, 10)
    p99 = np.percentile(volume, 99)
    volume = rescale_intensity(volume, in_range=(p10, p99))
    m = np.mean(volume, axis=(0, 1, 2))
    s = np.std(volume, axis=(0, 1, 2))
    volume = (volume - m) / s
    return volume

# Allow user to specify image size
parser = argparse.ArgumentParser()
parser.add_argument('--imsize',default=224,type=int,help='Desired image size')
args = parser.parse_args()

# Processing parameters
imsize = args.imsize
crop_thresh = 0.1

dataset_name = "kaggle_3m"
preprocessed_dataset_name = dataset_name + " - processed/"

# check for preprocessed folder, create if it doesn't exist
if os.path.isdir(preprocessed_dataset_name):
    print("\nThe processed dataset directory already exists. Delete or rename before running this script.")
else:
    os.mkdir(preprocessed_dataset_name)

# Go into data folder and get a list of patient directories
patient_dirs = glob(dataset_name+'/*/')

for patient_dir in patient_dirs:
    # Make the patient directory in the new folder
    new_patient_dir = preprocessed_dataset_name + patient_dir.replace(dataset_name,'')
    os.makedirs(new_patient_dir)
    # Get a list of image and mask filenames
    volume_names = glob(patient_dir+'*[0-9].tif')
    mask_names = glob(patient_dir+'*mask.tif')
    # Load the images and masks
    volumes = np.array([imread(volume_name) for volume_name in volume_names])
    masks = np.array([imread(mask_name, as_gray=True) for mask_name in mask_names])
    x = (volumes, masks)
    
    # Crop the images and masks to remove empty regions
    x_cropped = crop_sample(x,crop_thresh)
    # Pad the images and masks to be square
    x_padded = pad_sample(x_cropped)
    # Resize the images and masks
    x_resized = resize_sample(x_padded, size=imsize)
    # Normalize the image only, not mask
    #x_normalized = normalize_volume(x_resized[0])
    
    volumes_processed, masks_processed = x_resized
    for i in range(np.size(volumes_processed,0)):
        # Define the output paths
        image_path = new_patient_dir+os.path.basename(volume_names[i])
        mask_path = new_patient_dir+os.path.basename(mask_names[i])
        # Write the output files
        imsave(image_path,img_as_ubyte(volumes_processed[i]),check_contrast=False)
        imsave(mask_path,img_as_ubyte(masks_processed[i]),check_contrast=False)

# TODO  
# copy the other contents of the data folder into new folder

# output a txt file describing the parameters of the preprocessing

