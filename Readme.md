# Skin Disease Prediction

## Introduction
Skin disease has been hitting the headlines since the mankind has started exploiting the resources of planet earth hence creating the imbalances and in due process surging rise of pollutants in the atmosphere. The skin of most biotic organisms is gentle and it is the part that is mostly exposed leveraging high chances of catching the contaminated substances. It is these contaminated substances which attract harmful microorganisms like fungus, bacteria, etc., 
The chemical/physical reaction causes skin deformations in terms of colour and/(or) shape. 
It is often the case that we ignore the symptoms which in future may lead to carcinogenic activities inside skin cells. It takes seasoned eyes to identify the skin disease as there are overlapping symptoms that is deceptive of their exact type. 

We are proposing a computer vision solution that can be leveraged as a web application which when supplied with image of the person can predict the type of skin disease. It lends users with ability to perform a preliminary diagnosis of the skin inflammation or deformation which can be useful when a dermatologist is consulted. With this solution we can boil down the time spent in preliminary examination of the patients to identify the disease. This automated approach to skin disese identification can be put to regular scruitny by checking the performance on new set of test images to expand the horizon of diseses the system might identify correctly putting human in the loop. 

## Problem Statement
Given images, we have to come up with suitable computer vision model that can leverage the training performed on the images of disease samples to identify the most probable disease the provided image belongs into. 

## Proposed Solution
The population we are targeting in the solution is dominantly from India. There is no open sorce skin disease dataset for indian population due to which we have to manually collect and label the images which in our scenario ended up being implcitly balanced but sparse. This sparcity in the training samples is dealt through "Data Augmentation". This is neat jargon for the process of leveraging various image transformation techniques to create variations in terms of position, orientation, resolution and extent of zoom of Region of Interest in the images. Academic researchers have proved beyond the reasonable doubt that it is key technique in eliminating the chances of overfitting.
Apart from this image processing routine is used to extract out the skin from images calliberated for asian and african skin tones. 

This extracted skin image can then be fed into the machine learning model for inferencing so as to reduce the burden of processing unnecessary foreground and background pixels [pixels which are not part of human skin].

The Convolutional Neural Networks will be used as for inferencing due to their ability to capture spatial dependencies with translation and orientation invariant.

## Hardware requirements
	1. RAM >= 4GB
	2. AVX instruction set archiecture

## Software requirements
	1. Python 3.6–3.9; Python 3.9 support requires TensorFlow 2.5 or later; Python 3.8 support requires TensorFlow 2.2 or later.
	2. pip 19.0 or later (requires manylinux2010 support)
	3. Ubuntu 16.04 or later (64-bit)
	4. macOS 10.12.6 (Sierra) or later (64-bit) (no GPU support); macOS requires pip 20.3 or later
	5. Windows 7 or later (64-bit) ; Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017 and 2019
	6. GPU support requires a CUDA®-enabled card (Ubuntu and Windows)

## Data Collection
The source of images used in the training set is Google. A directed search has been done regarding images belonging to skin diseases dominantly reported in Indian patients. Following is source of the names of such skin diseases:
	1. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5029232/
	2. https://www.hindawi.com/journals/drp/2016/8608534/tab1/

Note : The design of the solution is extensively modular so any single module can be replaced with some variant without affecting other modules. Same applies with the data, As the requiement changes the dataset can be extended or completely changed to adapt to the ever changing requirements.

## Implementation
The code is segmented into five main components viz., Image Processing, Data Augmentation, Model architecture & operations, Model training and Inferencing. We will look at each module in brief bewlow.
	### Image Processing : The routines for image smoothing (noise reduction), Image sharpening (incraeses interpixel gradients to make edges crisp and clear), dynamic range adjustment (controls the brightness of the image) , Rotate and flip image on spatial axis.  
	### Data Augmentation : In start of this project we have marked our main challange and i.e. scarcity of data which has been fulfilled by morphing the small subset of image data (collected and labelled manually) based on image processing routines to have a variety in data points on which the model can train effectively.
	### Model Architecture : The CNN model architecture and various operations related to changing class integers into onehot vectors and vice versa apart from basic fit and predict methods are present.
	### Model Training : The Training of the model is done on the images stored locally in dataset folder which is passed through augmentation routine. The model is then saved as h5 file with the name specified in the properties file.
	### Inference engine : The images on which inferencing needs to be done is picked from processing directory which then is fed through image processing pipeline to extract the skin image on which the inferencing is done via the trained model.

## Setup 
run below command on the terminal to install the depencies. [Assuming python 3.x x>6 is already installed.]
$ python -m pip install -r requirements.txt

## Execution
Keep the images on which the inferencing needs to be done in ./static/processing_dir/ and then run the python script "inference.py" using -> $ python inference.py

