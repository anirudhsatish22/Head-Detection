SSD-based head detector
======

<div align="center">
    <img src="./examples/sample_detections.jpg" alt="Image showing head detections" height="302" width="404">
</div>

By Anirudh Satish, an extension of the work done by Pablo Medina-Suarez and Manuel J. Marin-Jimenez.

The following instructions for setting up the model follow directly from the Author's implementation. However, some parts of it are deprecated, and therefore I have included instructions that work as of May 2022. I have marked such sections with the tag Addition. 


This repository contains and showcases a head detector model for people detection in images. This model is based on 
the [Single Shot Multibox Detector (SSD)](https://arxiv.org/abs/1512.02325), as described in:
```
SSD: Single Shot MultiBox Detector
Authors: Liu, Wei; Anguelov, Dragomir; Erhan, Dumitru; Szegedy, Christian; Reed, Scott; Fu, Cheng-Yang; Berg, Alexander C. 
```

The model has been trained using the [Hollywood Heads dataset](https://www.robots.ox.ac.uk/~vgg/software/headmview/) as
positive samples, and a subsample of the [EgoHands dataset](http://vision.soic.indiana.edu/projects/egohands/) as negative
samples. This model has been developed using [Pierluigi Ferarri's Keras implementation of SSD](https://github.com/pierluigiferrari/ssd_keras/)
as primary source (of which we provide some essential code), and replicates the original [Matconvnet version of our model](https://github.com/AVAuco/ssd_people).


Quick start
------
### Cloning the repository
First, download a local copy of this repository, to do so, use the "Clone or download" button or run the following commands 
in a terminal:
```bash
# Install git:     
    sudo apt-get install git
# Clone ssd_head_keras from GitHub using the method of your choice: 
    git clone https://github.com/AVAuco/ssd_head_keras.git (HTTPS)
    git clone git@github.com:AVAuco/ssd_head_keras.git (SSH)
```

### Downloading the model
In the case you just want to download our detection model, we provide a ready to use version that you can download via 
the following links. Skip to the next section otherwise.

Since there are differences in the object serialization methods used between Python versions previous
to version 3.6, we provide two different versions of our model (we do not support Python 2.7):

- [Python versions below 3.6](https://drive.google.com/open?id=12cqKTPtQBAu780219hEbST7VwQuf6xDH).
- [Python versions above or equal to 3.6](https://drive.google.com/open?id=1vlmKOBtaT7eAd4_WcAv5MLBn7q_SWXoh) [(see Issue #22)](https://github.com/AVAuco/ssd_head_keras/issues/22)

In the `data` folder you can find a script that will download the model for you, just run the following commands:
```bash
# Install curl, if not already present
    sudo apt-get install curl
# Check your version of Python 3
    python3 --version
# Replace .Y for ".5" or ".6", depending on the output of the previous command 
    cd data
    chmod +x download_model_py3.Y.sh
# Run the script
    ./download_model_py3.Y.sh
```

Addition: As mentioned in the link above, the code does not work for Python versions 3.8 and later, as Tensorflow 1 is not supported on recent versions of Python.

### Setting up the model:
This section was added by me (Anirudh). While you are welcome to try setting up the model with the instructions provided by the author in the [how to use the model section](#how-to-use-the-model), I had limited success with that method due to deprecated modules and type incompatibilities. 


1. Install [Anaconda] (https://www.anaconda.com/) if you do not have it already 
2. Check that your python version is 3.7 or lower (but still python3)
3. Create a new Conda environment with the command "conda create -n <your environment name>
4. You can conda install or pip install the particular version of tensorflow required. I used pip with the following command "pip install tensorflow==1.15.0
5. Install Keras (version 2.2.4) with "pip install keras==2.2.4"
6. Downgrade numpy to version 1.16.4 if you have a more recent version to avoid deprecated function warnings.
7. Install matplotlib as I use this for image processing in certain places

Note that this method is more tedious than the virtual env route. I have updated the necessary versions in the requirements.txt file, so the latter method should also work should you wish to try. I just preder conda. 
Also note, that the [software requirements](#software-requirements) remain the same, with a GPU required to run the bounding box detection code. Evaluation and data processing scripts can be run on any machine with python3


### How to use the model (Author's version)
A brief tutorial is provided in the Jupyter notebook [demo_inference.ipynb](./demo_inference.ipynb). This tutorial 
explains how to use our model to detect heads over some example images.

To run this note book on your computer, first take a look at the [software requirements section](#software_reqs), then
run the following commands in a terminal:
```bash
# Activate the Python virtual environment
    source <venv_path>/bin/activate
# Set current directory to this repository's root path
    cd <download_path>/ssd_head_keras
# Start a notebook
    jupyter notebook
```
This command will launch a new tab on your default browser showing the Jupyter notebook environment, just click 
`demo_inference.ipynb` and follow the instructions in it.

Software requirements
------
<a id='software_reqs'></a>
These are the most relevant dependencies required to use our model [(see Issue #22)](https://github.com/AVAuco/ssd_head_keras/issues/22):
- Python packages: pip3, [virtualenv](https://virtualenv.pypa.io/en/latest/installation/) (recommended), 
[numpy](https://www.scipy.org/install.html#pip-install), [jupyterlab](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html) (tutorial), matplotlib, h5py.
- [Tensorflow](https://www.tensorflow.org/install/pip) (developed and tested on `tensorflow-gpu` 1.14).
- [Keras](https://keras.io/#installation) (developed and tested on version 2.2.4).
- [SSD Keras implementation](https://github.com/pierluigiferrari/ssd_keras) (essential code already provided in our repository).

Additional, recommended requirements to increase the inference performance on a NVIDIA GPU:
- NVIDIA CUDA Toolkit (tested on versions 9.0 and 10.0).
- **Optional:** a NVIDIA cuDNN version matching the NVIDIA CUDA Toolkit version installed.

An **optional, not recommended** [requirements file](./requirements.txt) is provided in this repository, which will allow you to install a new virtualenv
with all the required dependencies. Please **note** that this file has been used during development and may install additional, 
unnecessary packages in your system. If you opt for this option, run these commands in a terminal:
```bash
# Create a new Python 3 virtual environment
    virtualenv --system-site-packages -p python3 <venv_path>
# Activate the venv
    source <venv_path>/bin/activate
# Install this project dependencies using the provided requirements file
    pip install -r <download_path>/ssd_head_keras/requirements.txt
```

### Using the model (My version):
This section was added by me (Anirudh).

There are two main executable files that you can run, compute_frames.py and compute_coordinates.py

>**compute_frames.py**
This file, as expected can be used to predict/detect heads in frames, get the coordinates for the bounding boxes, AND create a new file in a folder 'outputData/{clip_name}', which must already exist for the code to run. Therefore, this program will create one clip per frame it is fed. You have the option to do this for either one directory (one clip), or a directory that has several sub directories of clips (with frames in those), depending on what you want. Use the argument parser to input your desired behaviour. 

>**compute_coordinates.py**
This script has the same behaviour as above, however instead of creating .png files with the drawn bounding boxes, it creates a .csv file per clip, where each row corresponds to one frame. The format of the file is as follows:
"frameName, box1_coordinates, box2_coordinates, ...."
If the frame has no coordinates, the row is of the form
"frameName"

Each box_coordinates cell/value is of the form [xmin, ymin, xmax, ymax]. With these coordinates, you can draw the boxes manually if you so please. Note that if frames contain multiple bounding boxes, that cooresponding row will have more columns, with each additional column being coordinates for one box in the specified format. 


### Performance
------

Addition:
If you wish to look at the performance of the authosr's implementation of the model, please refer to their [README.md](https://github.com/AVAuco/ssd_head_keras/blob/master/README.md)

For performance on the Gaze Data provided by Prof. Breeden, HMC, refer to the preliminary quantitative results plot in the Results folder, obtained by runnning evaluate.py


split_data.py was written to pre-process the labelled data for better evaluation!


Citation
------
```
@InProceedings{Marin19a,
    author       = "Marin-Jimenez, M.~J. and Kalogeiton, V. and Medina-Suarez, P. and Zisserman, A.",
    title        = "{LAEO-Net}: revisiting people {Looking At Each Other} in videos",
    booktitle    = "International Conference on Computer Vision and Pattern Recognition (CVPR)",
    year         = "2019",
}
```

For further questions or queries regarding this implementation of ssd_head_keras, contact me at asatish@hmc.edu 


Acknowledgements
------
Addition:
I thank the authors of both the MatConvNet and Tensorflow implementation of SSD based models, Liu, Wei; Anguelov, Dragomir; Erhan, Dumitru; Szegedy, Christian; Reed, Scott; Fu, Cheng-Yang; Berg, Alexander C. , Pablo Medina-Suarez and Manuel J. Marin-Jimenez. This work is nothing but an extension, and repurposing of the model already created and trained by these authors. I thank MIT for the licensing for this model, and making it publicly available and useable.  

I thank the authors of the images used in the demo code, which are licensed under a [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) license:
- [people_drinking.jpg](./examples/people_drinking.jpg), by [Ross Broadstock](https://www.flickr.com/people/figurepainting/).
- [rugby_players.jpg](./examples/rugby_players.jpg), by [jam_90s](https://www.flickr.com/people/zerospin/).
- [fish_bike.jpg](./examples/fish_bike.jpg), [source](https://github.com/BVLC/caffe/blob/master/examples/images/fish-bike.jpg).
