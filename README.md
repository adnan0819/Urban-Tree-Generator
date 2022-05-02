# Urban Tree Generator: Spatio-Temporal and Generative Deep Learning for Urban Tree Localization and Modeling 

This repository contains our dataset contribution and codebase to reproduce our paper titled **Urban Tree Generator: Spatio-Temporal and Generative Deep Learning for Urban Tree Localization and Modeling** in CGI 2022. Besides the codebase to reproduce our results, we hope that the dataset and codebase will help other researchers extend our methods in other domains also. 

# Annotated Dataset

As per our dataset contribution in our paper noted in Sec. 3.1, the annotated dataset of four cities (Chicago, Indianapolis, Austin, and Lagos) into three classes - tree, grass, others can be downloaded from [here](https://drive.google.com/file/d/1vP6ajrXvKzAcc8qvRUP3ciJAuZbWP392/view?usp=sharing).


A sample of the annotation of Indianapolis is shown below (green = tree, red = grass):

![annotation sample](Annotation_Sample.jpg "Annotation Sample over Indianapolis (green = tree, red = grass)")

# Codebase

## Requirements/Prerequisites for installation/training

All the required libraries are enlisted in ```requirements.txt```. To directly install using ```pip```, please just use:

```pip install -r requirements.txt```

## Segmentation and clustering

The repository is arranged so that can be easily reproducible into directories. The directory **Segmentation_and_clustering** contains all the code necessary to train and infer the segmentation and clustering section as noted in the paper. Here are some points as pre-requisites:


* Clone into this directory.
* Download the preprocessed training data from [here](https://drive.google.com/file/d/1sqaWBiUmq9EEZsVI9Cp1S_y4iRFx02uy/view?usp=sharing). 
* Place the zip file inside the ```Segmentation_and_clustering``` directory and unzip
* A directory called ```Data``` will be created
* Simply run ```./Segmentation_and_clustering/python train.py``` to train
* Inference and usage of pre-trained models are documented and commented inside ```train.py```

## Localization

The directory ```Localization``` contains all the code necessary to train and infer the localization  section as noted in the paper (Sec. 4). Here are some points as pre-requisites:


* Clone into this directory.
* Download the preprocessed training data from [here](https://drive.google.com/file/d/1HtExOOsc-o9ys7ZhNAWuvgRZYVMI4WJ7/view?usp=sharing). 
* Place the zip file inside the ```Localization``` directory and unzip
* Simply run ```./Localization/python train_localization.py``` to train the cGAN model
* Inference and usage of pre-trained models are documented and commented inside ```train_localization.py```. 

# Model Architectures

Below is our deep learning model for Segmentation of trees (Sec 3).

![segmentation model](48d_Segmentation_Model.png "48-dimensional Modified Unet Segmentation Model for Segmentation.")

For the model of our localization network (Sec. 4) please see the implementatation inside ```train_localization.py``` which is inspired by the standard Tensorflow [cGAN network](https://www.tensorflow.org/tutorials/generative/pix2pix_files/output_dIbRPFzjmV85_0.png). The figure is reproduced below.

![localization model](cGAN_Model.png "Conditional GAN Model for  Localization.")

Ref. Tensorflow [cGAN network](https://www.tensorflow.org/tutorials/generative/pix2pix_files/output_dIbRPFzjmV85_0.png). 

# Localization Output Example

An illustrative example of our Localization output (bottom row) and ground truth (top row) is shown below in a segment of Chicago (see Sec. 4 and Sec. 5 of our paper for more results and examples).

![Localization example](Localization_Example.jpg "Localization output (bottom row) and ground truth (top row).")
