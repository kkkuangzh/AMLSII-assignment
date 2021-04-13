# Train 4 models which use MSE and perceptual loss as loss functions and at scale of 2 and 4
# add file path of each task to system path
import sys
sys.path += ['../utility']
from help_functions import *
from extract_patches import *
from Network import *

import tensorflow as tf
from keras.models import load_model
from keras.optimizers import Adam
from keras.models import Model, Sequential
import keras


def prepare_patches(LR_train_unknown_path, HR_train_path, patch_height, patch_width, patch_num, up_scale):
    print('Extracting patches for training...')
    # load training images
    LR_train_imgs = load_data_from_dir(LR_train_unknown_path)
    HR_train_imgs = load_data_from_dir(HR_train_path)

    # extract 12000 image patches for training
    LR_patch_train, HR_patch_train = get_training_patches(LR_train_imgs, HR_train_imgs, patch_height, patch_width,
                                                          patch_num, up_scale)

    # normalize pixels to 0~1
    LR_patch_train = normalize(LR_patch_train)
    HR_patch_train = normalize(HR_patch_train)

    return LR_patch_train, HR_patch_train


# ---------------------------------- X2 model training --------------------------------------
# train model using MSE as loss function
def train_unknown_mse_x2_model(LR_patch_train, HR_patch_train, training_epoch):
    print('training unknown_mse_x2_model...')
    mse_model_x2 = final_model(48, 48, 3)
    mse_model_x2.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=['accuracy'])
    history = mse_model_x2.fit(LR_patch_train, HR_patch_train, epochs=training_epoch, verbose=1,
                                batch_size=4, validation_split=0.2)
    # mse_model_x2.save('./Track2/mse_unknown_X2.h5')
    return history


# train model using perceptual loss as loss function
def train_unknown_perceptual_x2_model(LR_patch_train, HR_patch_train, training_epoch):
    print('training unknown_perceptual_x2_model...')
    perceptual_loss_model_x2 = final_model(48, 48, 3)
    perceptual_loss_model_x2.compile(optimizer=Adam(lr=1e-4), loss=perceptual_loss_x2, metrics=['accuracy'])
    history = perceptual_loss_model_x2.fit(LR_patch_train, HR_patch_train, epochs=training_epoch, verbose=1,
                                            batch_size=4, validation_split=0.2)
    # perceptual_loss_model_x2.save('./Track2/perceptual_unknown_X2.h5')
    return history


def perceptual_loss_x2(y_true, y_pred):
    y_true = vgg_content_x2(y_true)
    y_predict = vgg_content_x2(y_pred)
    loss = keras.losses.mean_squared_error(y_true, y_predict)

    return loss


vgg_input, vgg_output = get_VGG19(input_size=(96, 96, 3))
vgg_content_x2 = Model(vgg_input, vgg_output)


# ---------------------------------- X4 model training --------------------------------------
# train model using MSE as loss function
def train_unknown_mse_x4_model(LR_patch_train, HR_patch_train, training_epoch):
    print('training unknown_mse_x4_model...')
    mse_model_x4 = build_mse_unknown_x4_model()
    # train unknown_X4 model
    mse_model_x4.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=['accuracy'])
    history = mse_model_x4.fit(LR_patch_train, HR_patch_train, epochs=training_epoch, verbose=1, batch_size=4,
                                validation_split=0.2)
    # mse_model_x4.save('./Track2/mse_unknown_X4.h5')
    return history


def build_mse_unknown_x4_model():
    # load trained MSE_X2 model
    x2_model = load_model('./Track2/mse_unknown_X2.h5', custom_objects={'tf': tf, 'perceptual_loss_x2': perceptual_loss_x2})
    # define the latter part of the integrated model
    x4_model = final_model(96, 96, 3)
    # cascade two models to achieve progressive super-resolution
    mse_model_x4 = integrated_network(x2_model, x4_model)

    return mse_model_x4


# train model using perceptual loss as loss function
def train_unknown_perceptual_x4_model(LR_patch_train, HR_patch_train, training_epoch):
    print('training unknown_perceptual_x4_model...')
    perceptual_model_x4 = build_perceptual_unknown_x4_model()

    # train unknown_X4 model
    perceptual_model_x4.compile(optimizer=Adam(lr=1e-4), loss=perceptual_loss_x4, metrics=['accuracy'])
    history = perceptual_model_x4.fit(LR_patch_train, HR_patch_train, epochs=training_epoch, verbose=1, batch_size=4,
                                       validation_split=0.2)
    # perceptual_model_x4.save('./Track2/perceptual_unknown_X4.h5')
    return history


def build_perceptual_unknown_x4_model():
    # load trained perceptual_X2 model
    x2_model = load_model('./Track2/perceptual_unknown_X2.h5',
                          custom_objects={'tf': tf, 'perceptual_loss_x2': perceptual_loss_x2})
    # define the latter part of the integrated model
    x4_model = final_model(96, 96, 3)
    perceptual_model_x4 = integrated_network(x2_model, x4_model)

    return perceptual_model_x4


# define the perceptual_loss_x4 for compare X4 model output of size (192, 192)
def perceptual_loss_x4(y_true, y_pred):
    y_t = vgg_content2(y_true)
    y_p = vgg_content2(y_pred)
    loss = keras.losses.mean_squared_error(y_t, y_p)

    return loss


vgg_input, vgg_output = get_VGG19(input_size=(192, 192, 3))
vgg_content2 = Model(vgg_input, vgg_output)








