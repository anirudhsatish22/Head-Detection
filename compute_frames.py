from platform import python_version
from PIL import Image
import argparse
import subprocess
import os

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

# Creating the Model with the required parameters
# This part follows directly from the ssd_head_keras code implementation


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
    '''This function takes in a directory for all the frames of a single clip, and a model as its only inputs
        Then, for each frame, it detects heads, and creates bounding boxes around it. Using the helper function, these bounding
        boxes are made on each frame and saved to a separate .png image'''
    all_files = list(os.walk(directory))
    clip = directory.split("/")[-1]

    frames = all_files[0][2]

    print(f"There are {len(frames)} frames for the movie clip {clip}")
    
    
    for picture in frames:
        full_path = directory + "/" + picture 
        locate_heads_single_frame(full_path, picture, clip)
       

def locate_heads_single_frame(full_path, picture, clip):
    ''' This function takes in a single frame,loads it and uses the model to predict the heads with bounding boxes. 
        It then creates a new .png image and draws the bounding box for the same, then saving this new image in a directory
        outputData/{movie_clip_name} that must already exist'''
    
    # Read the image and store as array of numbers, in a format that the model expects (in a list)
    input_images = [0] 
    orig_images = [0]
    orig_images[0] = image.load_img(full_path)

    # Reformat the input image to 512x512 ratio, as model expects this
    reformatted_image = image.load_img(full_path, target_size=(img_height, img_width)) 
    img = image.img_to_array(reformatted_image)
    input_images[0] =img

    input_images = np.array(input_images)

    # Get the predictions 
    y_pred = model.predict(input_images)


    confidence_threshold = 0.25

    # Perform confidence thresholding.
    y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]

    np.set_printoptions(precision=2, suppress=True, linewidth=90)


    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    classes = ['background', 'head']

    # Configure plot and disable axis
    # Draw the boxes

    fig = plt.figure(figsize=(8.54,3.7))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(orig_images[0])
    current_axis = plt.gca()

    # Display the image and draw the predicted boxes onto it.
    for box in y_pred_thresh[0]:
        # Transform the predicted bounding boxes for the 512x512 image to the original image dimensions.
        xmin = box[2] * np.array(orig_images[0]).shape[1] / img_width
        ymin = box[3] * np.array(orig_images[0]).shape[0] / img_height
        xmax = box[4] * np.array(orig_images[0]).shape[1] / img_width
        ymax = box[5] * np.array(orig_images[0]).shape[0] / img_height
        color = colors[int(box[0])]
        label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
        current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))
        current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})

    # Save the image in the required location
    new_name = "bounded" + picture
    new_path = 'outputData/' + clip + "/" + new_name
    plt.savefig(new_path)
    plt.close()


def compute_frames_for_all(root_directory, model):
    ''' Wrapper function to perform the detection for all movie clips '''

    list_of_movies = ['no_country_clip1','no_country_clip2', 'saving', 'shakespeare_clip1', 'shakespeare_clip2', 'slumdog', 'unforgiven']
    for movie in list_of_movies:
        print(f"running on movie {movie}")
        directory = root_directory+ movie
        find_boxes(directory, model)



def main(args):
    if (args.one_clip):    
        find_boxes(args.directory, model)
    elif (args.all_clips):
        compute_frames_for_all(args.directory, model)
    else:
        print(f"Please enter the required command line arguments")

    # If you just want to quickly test code, feel free to comment out the arg parser and use the following function calls
    # With different paths as suited to you. 

    # root_directory = '/ocean/projects/cis220010p/shared/frames/argo'
    # find_boxes(root_directory, model) 

    # or
    # root_directory = '/ocean/projects/cis220010p/shared/frames/argo'
    # compute_frames_for_all(root_directory, model)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    one_or_all = parser.add_mutually_exclusive_group(required=True)
    one_or_all.add_argument("--one_clip", action="store_true")
    one_or_all.add_argument("--all_clips", action="store_true")

    parser.add_argument("--directory", help="Directory name. Depends on previous mutually exclusive arguemnts. \
                            directory is either directory of all clips, or one clip")
    args=parser.parse_args()
    main(args)
