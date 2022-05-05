from platform import python_version
from PIL import Image
import subprocess
import os
import csv
from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam
import numpy as np
from matplotlib import pyplot as plt

from models.keras_ssd512 import ssd_512
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization


model_version = '3.6'
img_height = 512 
img_width = 512 

# Set the model's inference mode
model_mode = 'inference'


# Set the desired confidence threshold
conf_thresh = 0.01

# 1: Build the Keras model
K.clear_session() # Clear previous models from memory.
model = ssd_512(image_size=(img_height, img_width, 3),
                n_classes=1,
                mode=model_mode,
                l2_regularization=0.0005,
                scales=[0.07, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05], # PASCAL VOC
                aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5]],
               two_boxes_for_ar1=True,
               steps=[8, 16, 32, 64, 128, 256, 512],
               offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
               clip_boxes=False,
               variances=[0.1, 0.1, 0.2, 0.2],
               normalize_coords=True,
               subtract_mean=[123, 117, 104],
               swap_channels=[2, 1, 0],
               confidence_thresh=conf_thresh,
               iou_threshold=0.45,
               top_k=200,
               nms_max_output_size=400)

# 2: Load the trained weights into the model. Make sure the path correctly points to the model's .h5 file
weights_path = './data/ssd512-hollywood-trainval-bs_16-lr_1e-05-scale_pascal-epoch-187-py%s.h5' % model_version
model.load_weights(weights_path, by_name=True)

# 3: Compile the model so that Keras won't complain the next time you load it.
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

def find_boxes(directory, model):
    all_files = list(os.walk(directory))
    clip = directory.split("/")[-1]

    frames = all_files[0][2]

    print(f"There are {len(frames)} frames for the movie clip {clip}")
    
    output_path = "/ocean/projects/cis220010p/asatish/ssd_head_keras/outputCsv/" + clip + ".csv" 
    outputFP = open(output_path, 'w')
    outputWriter = csv.writer(outputFP)
    outputWriter.writerow(['frame', 'boxes'])
    
    for picture in frames:
        full_path = directory + "/" + picture 
        helper(full_path, picture, clip, outputWriter)
    outputFP.close()
       

def helper(full_path, picture, clip, outputWriter):
    input_images = [0] 
    orig_images = [0]
    orig_images[0] = image.load_img(full_path)
    reformatted_image = image.load_img(full_path, target_size=(img_height, img_width)) 
    img = image.img_to_array(reformatted_image)
    input_images[0] =img

    input_images = np.array(input_images)
    y_pred = model.predict(input_images)


    confidence_threshold = 0.25

    # Perform confidence thresholding.
    y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]

    np.set_printoptions(precision=2, suppress=True, linewidth=90)


    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    classes = ['background', 'head']

    newRow = [picture]
    # Display the image and draw the predicted boxes onto it.
    for box in y_pred_thresh[0]:
        # Transform the predicted bounding boxes for the 512x512 image to the original image dimensions.
        xmin = box[2] * np.array(orig_images[0]).shape[1] / img_width
        ymin = box[3] * np.array(orig_images[0]).shape[0] / img_height
        xmax = box[4] * np.array(orig_images[0]).shape[1] / img_width
        ymax = box[5] * np.array(orig_images[0]).shape[0] / img_height
        newRow.append([xmin, ymin, xmax, ymax])


    outputWriter.writerow(newRow)

    print(picture)


def compute_frames_for_all(root_directory, model):
    list_of_movies1 = ['amadeus', 'argo', 'birdman', 'chicago', 'departed', 'emperor', 'kings', 'gladiator']
    list_of_movies2 = ['no_country_clip1','no_country_clip2', 'saving', 'shakespeare_clip1', 'shakespeare_clip2', 'slumdog', 'unforgiven']
    for movie in list_of_movies2:
        print(f"running on movie {movie}")
        directory = root_directory+ movie
        find_boxes(directory, model)

root_directory = '/ocean/projects/cis220010p/shared/frames/'
compute_frames_for_all(root_directory, model)
