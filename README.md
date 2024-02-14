# Temporal Action Detection based on Action Former
This code based on Action former.
The code is based on the feature pyramid and consists of blocks in the pooling layer and blocks in the convolution layer

*[paper link](https://arxiv.org/abs/2202.07925)
*[code link](https://github.com/happyharrycn/actionformer_release)

## Code Overview
The structure of this code repo is heavily inspired by Detectron2. Some of the main components are
* ./libs/core: Parameter configuration module.
* ./libs/datasets: Data loader and IO module.
* ./libs/modeling: Our main model with all its building blocks.
* ./libs/utils: Utility functions for training, inference, and postprocessing.

## Installation
* Follow INSTALL.md for installing necessary dependencies and compiling the code.

**Download Features and Annotations**
* Download *thumos.tar.gz* (`md5sum 375f76ffbf7447af1035e694971ec9b2`) from [this Box link](https://uwmadison.box.com/s/glpuxadymf3gd01m1cj6g5c3bn39qbgr) or [this Google Drive link](https://drive.google.com/file/d/1zt2eoldshf99vJMDuu8jqxda55dCyhZP/view?usp=sharing) or [this BaiduYun link](https://pan.baidu.com/s/1TgS91LVV-vzFTgIHl1AEGA?pwd=74eh).
* The file includes I3D features, action annotations in json format (similar to ActivityNet annotation format), and external classification scores.

* **Unpack Features and Annotations**
* Unpack the file under *./data* (or elsewhere and link to *./data*).
* The folder structure should look like
```
This folder
│   README.md
│   ...  
│
└───data/
│    └───thumos/
│    │	 └───annotations
│    │	 └───i3d_features   
│    └───...
|
└───libs
│
│   ...
```


**Unpack Features and Annotations**
* Unpack the file under *./data* (or elsewhere and link to *./data*).
* The folder structure should look like
```
This folder
│   README.md
│   ...  
│
└───data/
│    └───anet_1.3/
│    │	 └───annotations
│    │	 └───tsp_features   
│    └───...
|
└───libs
│
│   ...

