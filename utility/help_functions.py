import numpy as np
import os
from PIL import Image
from extract_patches import *
from keras.applications.vgg19 import VGG19
from keras.layers import Input
from keras.models import Sequential


# normalize images from 0~255 to 0~1
def normalize(imgs):
    return imgs / 255


def denormalize(imgs):
    imgs = imgs * 255
    return imgs.astype(np.uint8)


# prepare images from directory
def load_data_from_dir(img_path):
    res_list = []
    for path, subpath, files in os.walk(img_path):
        files.sort()
        for i in files:
            if i == '.DS_Store':
                continue
            img = Image.open(img_path + i)
            res_list.append(np.asarray(img))
    print("loaded image number: ", len(res_list))

    return res_list


# randomly crop pathches from training images
# return type is uint8 so that the image can be visualized
def get_training_patches(lr_imgs, hr_imgs, patch_h, patch_w, patchnum, scale):
    lr_patch, hr_patch = train_patch(lr_imgs, hr_imgs, patch_h, patch_w, patchnum, scale)
    print("LR_patch shape: ", lr_patch.shape)
    print("HR_patch shape: ", hr_patch.shape)

    return lr_patch, hr_patch


# define perceptual loss based on the first 5 layers of VGG19 model
def get_VGG19(input_size):
    vgg_input = Input(input_size)
    vgg = VGG19(include_top=False, input_tensor=vgg_input)
    for l in vgg.layers:
        l.trainable = False
    vgg_output = vgg.get_layer('block2_conv2').output

    return vgg_input, vgg_output


# cascade two models to achieve progressive super-resolution
# set the first part of model not trainable
def integrated_network(base_model1, base_model2):
    base_model1.trainable = False
    add_model = Sequential()
    add_model.add(base_model1)
    add_model.add(base_model2)

    return add_model