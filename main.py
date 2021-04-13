# add file path of each task to system path
import sys
sys.path += ['./Track1/', './Track2/', './utility']
from extract_patches import *
from help_functions import *
from Network import *
from bicubic_models import *
from bicubic_evaluation import *
from unknown_models import *
from unknown_evaluation import *
import numpy as np

# set parameters
patch_height = 48
patch_width = 48
patch_num = 40
training_epoch = 2
HR_train_path = './Datasets/DIV2K_train_HR/'


# -----------------------------------------------------------------------------------------------------
# ---------------------------------- Track1: bicubic degradation --------------------------------------
# -----------------------------------------------------------------------------------------------------
# X2 up-scaling
up_scale = 2
LR_train_bicubic_path = './Datasets/DIV2K_train_LR_bicubic/X2/'

# load 12000 bicubic training patches pairs of (48,48), (96, 96)
LR_patch_train, HR_patch_train = prepare_patches(LR_train_bicubic_path, HR_train_path,
                                                 patch_height, patch_width, patch_num, up_scale)
# train models
history1 = train_bi_mse_x2_model(LR_patch_train, HR_patch_train, training_epoch)
history2 = train_bi_perceptual_x2_model(LR_patch_train, HR_patch_train, training_epoch)


# X4 up-scaling
up_scale = 4
LR_train_bicubic_path = './Datasets/DIV2K_train_LR_bicubic/X4/'

# load 12000 bicubic training patches pairs of (48,48), (192, 192)
LR_patch_train, HR_patch_train = prepare_patches(LR_train_bicubic_path, HR_train_path,
                                                 patch_height, patch_width, patch_num, up_scale)
# train models
history3 = train_bi_mse_x4_model(LR_patch_train, HR_patch_train, training_epoch)
history4 = train_bi_perceptual_x4_model(LR_patch_train, HR_patch_train, training_epoch)


# -----------------------------------------------------------------------------------------------------
# ---------------------------------- Track2: unknown degradation --------------------------------------
# -----------------------------------------------------------------------------------------------------
# X2 up-scaling
up_scale = 2
LR_train_unknown_path = './Datasets/DIV2K_train_LR_unknown/X2/'

# load 12000 unknown training patches pairs of (48,48), (96, 96)
LR_patch_train, HR_patch_train = prepare_patches(LR_train_unknown_path, HR_train_path,
                                                 patch_height, patch_width, patch_num, up_scale)
# train models
history5 = train_unknown_mse_x2_model(LR_patch_train, HR_patch_train, training_epoch)
history6 = train_unknown_perceptual_x2_model(LR_patch_train, HR_patch_train, training_epoch)


# X4 up-scaling
up_scale = 4
LR_train_unknown_path = './Datasets/DIV2K_train_LR_unknown/X4/'

# load 12000 unknown training patches pairs of (48,48), (192, 192)
LR_patch_train, HR_patch_train = prepare_patches(LR_train_unknown_path, HR_train_path,
                                                 patch_height, patch_width, patch_num, up_scale)
# train models
history7 = train_unknown_mse_x4_model(LR_patch_train, HR_patch_train, training_epoch)
history8 = train_unknown_perceptual_x4_model(LR_patch_train, HR_patch_train, training_epoch)


# -----------------------------------------------------------------------------------------------------
# ------------------------------------------- Evaluation ----------------------------------------------
# -----------------------------------------------------------------------------------------------------
HR_valid_path = './Datasets/DIV2K_valid_HR/'
model_path1 = './Models/mse_bicubic_X2.h5'
model_path2 = './Models/mse_bicubic_X4.h5'
model_path3 = './Models/mse_unknown_X2.h5'
model_path4 = './Models/mse_unknown_X4.h5'
model_path5 = './Models/perceptual_bicubic_X2.h5'
model_path6 = './Models/perceptual_bicubic_X4.h5'
model_path7 = './Models/perceptual_unknown_X2.h5'
model_path8 = './Models/perceptual_unknown_X4.h5'

test_num = 2
stride = 40

# bicubic X2
LR_valid_bicubic_path_x2 = './Datasets/DIV2K_valid_LR_bicubic/X2/'
PSNR_mse_bi_x2, SSIM_mse_bi_x2 = \
    evaluate_on_DIV2K_bi(LR_valid_bicubic_path_x2, HR_valid_path, model_path1, test_num, stride, 2)
PSNR_perceptual_bi_x2, SSIM_perceptual_bi_x2 = \
    evaluate_on_DIV2K_bi(LR_valid_bicubic_path_x2, HR_valid_path, model_path5, test_num, stride, 2)

# unknown X2
LR_valid_unknown_path_x2 = './Datasets/DIV2K_valid_LR_unknown/X2/'
PSNR_mse_unknown_x2, SSIM_mse_unknown_x2 = \
    evaluate_on_DIV2K_unknown(LR_valid_unknown_path_x2, HR_valid_path, model_path3, test_num, stride, 2)
PSNR_perceptual_unknown_x2, SSIM_perceptual_unknown_x2 = \
    evaluate_on_DIV2K_unknown(LR_valid_unknown_path_x2, HR_valid_path, model_path7, test_num, stride, 2)

# bicubic X4
LR_valid_bicubic_path_x4 = './Datasets/DIV2K_valid_LR_bicubic/X4/'
PSNR_mse_bi_x4, SSIM_mse_bi_x4 = \
    evaluate_on_DIV2K_bi(LR_valid_bicubic_path_x4, HR_valid_path, model_path2, test_num, stride, 4)
PSNR_perceptual_bi_x4, SSIM_perceptual_bi_x4 = \
    evaluate_on_DIV2K_bi(LR_valid_bicubic_path_x4, HR_valid_path, model_path6, test_num, stride, 4)

# unknown X4
LR_valid_unknown_path_x4 = './Datasets/DIV2K_valid_LR_unknown/X4/'
PSNR_mse_unknown_x4, SSIM_mse_unknown_x4 = \
    evaluate_on_DIV2K_unknown(LR_valid_unknown_path_x4, HR_valid_path, model_path4, test_num, stride, 4)
PSNR_perceptual_unknown_x4, SSIM_perceptual_unknown_x4 = \
    evaluate_on_DIV2K_unknown(LR_valid_unknown_path_x4, HR_valid_path, model_path8, test_num, stride, 4)


# -----------------------------------------------------------------------------------------------------
# ---------------------------------- Evaluation on Set5 and Set14 -------------------------------------
# -----------------------------------------------------------------------------------------------------
Set5 = './Datasets/Set5/'
Set5_x2 = './Datasets/Set5_x2/'
Set5_x4 = './Datasets/Set5_x4/'
PSNR_set5_x2, SSIM_set5_x2 = evaluate_on_Set5_Set14(Set5_x2, Set5, model_path1, test_num=2, stride=30, up_scale=2)
PSNR_set5_x4, SSIM_set5_x4 = evaluate_on_Set5_Set14(Set5_x4, Set5, model_path2, test_num=2, stride=30, up_scale=4)

Set14 = './Datasets/Set14/'
Set14_x2 = './Datasets/Set14_x2/'
Set14_x4 = './Datasets/Set14_x4/'
PSNR_set14_x2, SSIM_set14_x2 = evaluate_on_Set5_Set14(Set14_x2, Set14, model_path1, test_num=2, stride=30, up_scale=2)
PSNR_set14_x4, SSIM_set14_x4 = evaluate_on_Set5_Set14(Set14_x4, Set14, model_path2, test_num=2, stride=30, up_scale=4)


# -----------------------------------------------------------------------------------------------------
# ------------------------------------------- Print results -------------------------------------------
# -----------------------------------------------------------------------------------------------------
print('Mean PSNR and SSIM on DIV2K validation set:')
print('Bicubic_mse_x2: PSNR', np.mean(PSNR_mse_bi_x2), ', SSIM', np.mean(SSIM_mse_bi_x2))
print('Bicubic_mse_x4: PSNR', np.mean(PSNR_mse_bi_x4), ', SSIM', np.mean(SSIM_mse_bi_x4))
print('Bicubic_perceptual_x2: PSNR', np.mean(PSNR_perceptual_bi_x2), ', SSIM', np.mean(SSIM_perceptual_bi_x2))
print('Bicubic_perceptual_x4: PSNR', np.mean(PSNR_perceptual_bi_x4), ', SSIM', np.mean(SSIM_perceptual_bi_x4))
print('Unknown_mse_x2: PSNR', np.mean(PSNR_mse_unknown_x2), ', SSIM', np.mean(SSIM_mse_unknown_x2))
print('Unknown_mse_x4: PSNR', np.mean(PSNR_mse_unknown_x4), ', SSIM', np.mean(SSIM_mse_unknown_x4))
print('Unknown_perceptual_x2: PSNR', np.mean(PSNR_perceptual_unknown_x2), ', SSIM', np.mean(SSIM_perceptual_unknown_x2))
print('Unknown_perceptual_x4: PSNR', np.mean(PSNR_perceptual_unknown_x4), ', SSIM', np.mean(SSIM_perceptual_unknown_x4), '\n')

print('Mean PSNR and SSIM on Set5:')
print('Set5_x2: PSNR', np.mean(PSNR_set5_x2), ', SSIM', np.mean(SSIM_set5_x2))
print('Set5_x4: PSNR', np.mean(PSNR_set5_x4), ', SSIM', np.mean(SSIM_set5_x4), '\n')
print('Mean PSNR and SSIM on Set14:')
print('Set14_x2: PSNR', np.mean(PSNR_set14_x2), ', SSIM', np.mean(SSIM_set14_x2))
print('Set14_x4: PSNR', np.mean(PSNR_set14_x4), ', SSIM', np.mean(SSIM_set14_x4))