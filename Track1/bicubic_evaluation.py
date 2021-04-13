
import sys
sys.path += ['../utility']
from help_functions import *
from extract_patches import *
from Network import *

import tensorflow as tf
from keras.models import load_model
from keras.models import Model
import keras
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def load_bi_test_images(LR_valid_bicubic_path, HR_valid_path):
    print('Loading test images...')
    LR_valid_imgs = load_data_from_dir(LR_valid_bicubic_path)
    HR_valid_imgs = load_data_from_dir(HR_valid_path)

    return LR_valid_imgs, HR_valid_imgs


def evaluate_on_DIV2K_bi(LR_valid_bicubic_path, HR_valid_path, model_path, test_num, stride, up_scale):
    LR_valid_imgs, HR_valid_imgs = load_bi_test_images(LR_valid_bicubic_path, HR_valid_path)
    model = load_model(model_path, custom_objects={'tf': tf, 'perceptual_loss_x2': perceptual_loss_x2,
                                                   'perceptual_loss_x4': perceptual_loss_x4})
    for i in range(len(LR_valid_imgs)):
        LR_valid_imgs[i] = normalize(LR_valid_imgs[i])
    
    print('Predicting test images...')
    predicted_HR_list = test_patch(LR_valid_imgs, test_num, 48, 48, stride, model, up_scale)
    for i in range(len(predicted_HR_list)):
        predicted_HR_list[i] = denormalize(predicted_HR_list[i])

    print('Evaluating test images...')
    PSNR_val, SSIM_val = calculate_psnr_ssim_bi(predicted_HR_list, HR_valid_imgs)
    return PSNR_val, SSIM_val


def calculate_psnr_ssim_bi(predicted_HR_list, HR_valid_imgs):
    PSNR_val = []
    SSIM_val = []

    for i in range(len(predicted_HR_list)):
        PSNR = peak_signal_noise_ratio(HR_valid_imgs[i], predicted_HR_list[i])
        SSIM = structural_similarity(HR_valid_imgs[i], predicted_HR_list[i], multichannel=True)
        PSNR_val.append(PSNR)
        SSIM_val.append(SSIM)

    return PSNR_val, SSIM_val


def perceptual_loss_x2(y_true, y_pred):
    y_true = vgg_content_x2(y_true)
    y_predict = vgg_content_x2(y_pred)
    loss = keras.losses.mean_squared_error(y_true, y_predict)

    return loss


def perceptual_loss_x4(y_true, y_pred):
    y_t = vgg_content_x4(y_true)
    y_p = vgg_content_x4(y_pred)
    loss = keras.losses.mean_squared_error(y_t, y_p)

    return loss


vgg_input, vgg_output = get_VGG19(input_size=(96, 96, 3))
vgg_content_x2 = Model(vgg_input, vgg_output)
vgg_input, vgg_output = get_VGG19(input_size=(192, 192, 3))
vgg_content_x4 = Model(vgg_input, vgg_output)


def evaluate_on_Set5_Set14(LR_valid_bicubic_path, HR_valid_path, model_path, test_num, stride, up_scale):
    LR_valid_imgs, HR_valid_imgs = load_bi_test_images(LR_valid_bicubic_path, HR_valid_path)
    model = load_model(model_path, custom_objects={'tf': tf})
    for i in range(len(LR_valid_imgs)):
        LR_valid_imgs[i] = normalize(LR_valid_imgs[i])

    print('Predicting test images...')
    predicted_HR_list = test_patch(LR_valid_imgs, test_num, 48, 48, stride, model, up_scale)
    for i in range(len(predicted_HR_list)):
        predicted_HR_list[i] = denormalize(predicted_HR_list[i])

    print('Evaluating test images...')
    PSNR_val, SSIM_val = calculate_psnr_ssim_bi(predicted_HR_list, HR_valid_imgs)
    return PSNR_val, SSIM_val








