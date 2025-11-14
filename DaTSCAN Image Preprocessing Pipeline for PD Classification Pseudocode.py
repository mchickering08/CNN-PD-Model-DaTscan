import os #working w/ file paths and directory listings
import numpy as np #numerical arrays and math operations
import pandas as pd #reading metadata tables (CSV)
import pydicom #loading DICOM med imaging files !
import SimpleITK as sitk #for image resampling and med image processing !!
import cv2 #OpenCV for resizing images and basic image operations !!

from sklearn.model_selection import train_test_split #creating train/val/test splits

import tensorflow as tf #dl framework
from tensorflow.keras.applications import EfficientNetB0 # type: ignore #EfficientNet-B0 architecture
from tensorflow.keras import layers, models  # type: ignore #layers = building blocks, models = Model() class for NN architecture

#define paths to data
ppmi_metadata_csv = "path/xyz.csv" #csv with subject id, classification
ppmi_dicom_holder = "path/dicom" #root folder w/ one sub folder per subject
out_dataset_path = "datscan_dataset.npz" #save processed data arrays (numpy file format)

#define image processing hyperparameters
target_spacing = (2.0, 2.0, 2.0) #example spacing in mm (x, y, z)
crop_xy_size = 80 #side of center crop in x and y (voxels)
crop_z_min = 15 #lower index limit for z axis (should be mid brain start)
crop_z_max = 45 #upper index limit for z axis (mid brain end)
img_size = (224, 224) #final 2d spatial size for EN input
batch_size = 16 #batch size for training data (subject to change)
epochs = 20 #max number of training epochs (also subject to change)