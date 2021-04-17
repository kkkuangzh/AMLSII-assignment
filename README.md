
# AMLS II assignment
---
You can get eveything that's needed for the assignment from [here](https://drive.google.com/drive/folders/186Q3Q9QyNmPO6SVMpC6lfanDtHnITuU6?usp=sharing).

Note that github repository **does not** contain all the files due to size limits.


*All the experiments of this assignment are carried out using jupyter notebook in local laptop or in the server. In order to keep folders neat and clean, codes are reorganized to python files with necessary refactoryed functions. Some ipynb files are also provided with expected output and annotations.*

## Problem description

This assignment proposes a CNN model architecture to deal with super-resolution problem. The dataset used in this experiment are from [NTIRE2017 challenge](https://data.vision.ee.ethz.ch/cvl/DIV2K/) which includes 800 training images and 100 validation images. 

Images from Track1 and Track2 of scale X2 and X4 are used to train and evaluate the models.

Additionally, [Set5](http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html) and [Set14](https://paperswithcode.com/dataset/set14) datasets are also used to evaluate and compare performances with other methods.



## Running Environment
---
All the codes are written in python 3.6.8 environment and experimented on Linux servers and Mac OS. 

Some packetages needed for this task are given here long with its version. 
*tensorflow is not the newest version due to server constraints*

+ keras 2.3.1
+ numpy 1.19.5
+ Pillow 8.1.2
+ tensorflow 2.1.0
+ scikit-image 0.17.2



## To run 
---
**The images used for each task should be copied into the correct folder under the Dataset folder first.**

If everything is downloaded and installed, you can simply run the main.py to train models and get the results.

The model selection process should be run separately from ./Model selection/model selection.py

An additional main.ipynb is provided with expected output, you can also run this file as each section can be run separately.

## Attentions

The default settings are set to be small (e.g., patch_num, training_epoch, test_num etc.) to demonstrate rather than retraining the models.

All the *checkpointer* and *model.save()* are commented out as the process is shown on corresponding ipynb files. If you want to wait and retrain a model, just uncomment them, but remember to change the name and path as it may cover the pretrained model.

These ipynb files provided in each folders are also organized and annotated to save time running py files. However, the codes are not refactoryed and as they are run in different environment, the paths may not be applicable in this setting. These ipynb files (except main.ipynb) are provided to show the expected training process and outcome.

Using the main.ipynb file is more convienient to train and stop training a model.

## Project Organization
---

### Base folder
---
* main.py: used to run the whole project.
* main.ipynb: same as main.py, only with expected results and can be run separately for each section.
* Datasets: Images from Set5 and Set14 are provided. Images for DIV2K should be uploaded first to six sub-folders.
* Track 1 and Track 2 folders contain model training and evaluation functions along with model files used for each task.
* Model selection folder contains code for training and testing modifications based on the baseline model.
* utility folder contains necessary functions.
* images folder contains reconstruct images of bicubic interpolation and two models.

### Folder Track 1
---
* mse_bicubic_X2.h5: model trained using MSE dealing with images of bicubic down-sampling at scale 2
* mse_bicubic_X4.h5: model trained using MSE dealing with images of bicubic down-sampling at scale 4
* perceptual_bicubic_X2.h5: model trained using perceptual loss dealing with images of bicubic down-sampling at scale 2
* perceptual_bicubic_X4.h5: model trained using perceptual loss dealing with images of bicubic down-sampling at scale 4
* bicubic_evaluation.py: contains functions related to 
  - load training images from directory,
  - recover from test patches and compare with HR images,
  - calculate PSNR and SSIM metrics of models.

* bicubic_model.py: contains 
  - extraction of training patches from bicubic images,
  - build X4 model by cascading two X2 models,
  - define perceptual loss,
  - training functions of four models.

* Model training(X2).ipynb: contains 
  - the actual training process of X2 models.
* Model training(X4).ipynb: contains 
  - the actual training process of X4 models.
* Evaluation_mse.ipynb: contains 
  - the testing time of different strides of models,
  - the PSNR and SSIM metrics of models using MSE as loss function on DIV2K validation set.
* Evaluation_Set5 and Set14.ipynb: contains 
  - the PSNR and SSIM metrics of models using MSE as loss function on Set5 and Set14,
  - visulised patches of different models.

The *model.save( )* have been commented out to avoid replacing the trained models.

learning rate = 1e-4 and batch size = 4 are set by default.


### Folder Track 2
---
* mse_unknown_X2.h5: model trained using MSE dealing with images of unknown down-sampling at scale 2
* mse_unknown_X4.h5: model trained using MSE dealing with images of unknown down-sampling at scale 4
* perceptual_unknown_X2.h5: model trained using perceptual loss dealing with images of unknown down-sampling at scale 2
* perceptual_unknown_X4.h5: model trained using perceptual loss dealing with images of unknown down-sampling at scale 4
* unknown_evaluation.py: contains functions related to 
  - load training images from directory,
  - recover from test patches and compare with HR images,
  - calculate PSNR and SSIM metrics of models.

* unknown_model.py: contains 
  - extraction of training patches from unknown images,
  - build X4 model by cascading two X2 models,
  - define perceptual loss,
  - training functions of four models.

* Evaluation_perceptual.ipynb: contains 
  - the testing time of different strides of models,
  - the PSNR and SSIM metrics of models using MSE as loss function on DIV2K validation set.

The *model.save( )* have been commented out to avoid replacing the trained models.


### Folder Model selection
---
* model_selection.py: contains functions related to
  - load images from directory,
  - define and training of 7 different models,
  - evaluate trained models on 10 test images.
* model_selection.ipynb: contains functions related to
  - the actual training process of 7 different models.

### Folder utility
---
* extract_patches.py: contains functions related to
  - extract patches randomly for training,
  - extract overlap patches for testing,
  - recombine patches to full image size.
  
* help_function.py: contains functions related to
  - load images from directory,
  - normalize data,
  - define VGG model,
  - cascade two X2 models.
  
* Network.py: contains functions of related to
  - define subpixel layer for up-sampling,
  - define residual block used to build network,
  - build three network architectures for model comparison,
  - build the final network used in Track 1 and 2.


