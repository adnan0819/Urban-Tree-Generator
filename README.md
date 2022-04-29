# Urban Tree Generator: Spatio-Temporal and Generative Deep Learning for Urban Tree Localization and Modeling 

This repository contains our dataset contribution and codebase to reproduce our paper titled **Urban Tree Generator: Spatio-Temporal and Generative Deep Learning for Urban Tree Localization and Modeling** for CGI 2022. Besides the codebase to reproduce our results, we hope that the dataset and codebase will help other researchers extend our methods in other domains also. 

# Annotated Dataset

As per our dataset contribution in our paper noted in Sec. 3.1, the annotated dataset of four cities (Chicago, Indianapolis, Austin, and Lagos) into three classes - tree, grass, others can be downloaded from **redacted for anonymity**.

# Codebase

## Segmentation and Clustering

The repository is arranged so that can be easily reproducible into directories. The directory **Segmentation_and_clustering** contains all the code necessary to train and infer the segmentation and clustering section as noted in the paper. Here are some points as pre-requisites:

* Download the preprocessed training data from **redacted for anonymity**
* Place the zip file inside the **Segmentation_and_clustering** directory and unzip
* A directory called **Data** will be created
* Simply run train.py to train
* Inference and usage of pre-trained models are commented inside train.py

## Localization

The directory **Localization** contains all the code necessary to train and infer the localization  section as noted in the paper (Sec. 4). Here are some points as pre-requisites:

* Download the preprocessed training data from **redacted for anonymity**
* Place the zip file inside the **Localization** directory and unzip
* Simply run train_localization.py to train the cGAN model
* Inference and usage of pre-trained models are commented inside train_localization.py

With [Handlebars templates](http://handlebarsjs.com/),

