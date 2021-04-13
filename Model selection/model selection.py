# add file path of each task to system path
import sys
sys.path += ['../utility']
from help_functions import *
from extract_patches import *
from Network import *

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from keras.models import load_model
from keras.applications.vgg19 import VGG19
from keras.layers import Input
from keras.optimizers import Adam
from keras.models import Model
import keras

# set parameters
LR_train_bicubic_path = '../Datasets/DIV2K_train_LR_bicubic/X2/'
LR_valid_bicubic_path = '../Datesets/DIV2K_valid_LR_bicubic/X2/'
HR_train_path = '../Datasets/DIV2K_train_HR/'
HR_valid_path = '../Datasets/DIV2K_valid_HR/'

up_scale = 2
patch_height = 48
patch_width = 48
patch_num = 40

# load images
LR_train_imgs = load_data_from_dir(LR_train_bicubic_path)
HR_train_imgs = load_data_from_dir(HR_train_path)

# get image patches for training
LR_patch_train, HR_patch_train = get_training_patches(LR_train_imgs, HR_train_imgs, patch_height, patch_width, patch_num, up_scale)

# normalize pixels to 0~1
LR_patch_train = normalize(LR_patch_train)
HR_patch_train = normalize(HR_patch_train)


# define perceptual loss based on the first 5 layers of VGG19 model
def get_VGG19(input_size):
    vgg_input = Input(input_size)
    vgg = VGG19(include_top=False, input_tensor=vgg_input)
    for l in vgg.layers:
        l.trainable = False
    vgg_output = vgg.get_layer('block2_conv2').output

    return vgg_input, vgg_output


def perceptual_loss(y_true, y_pred):
    y_true = vgg_content(y_true)
    y_predict = vgg_content(y_pred)
    loss = keras.losses.mean_squared_error(y_true, y_predict)

    return loss


vgg_input, vgg_output = get_VGG19(input_size=(96, 96, 3))
vgg_content = Model(vgg_input, vgg_output)


# -----------------------------------------------------------------------------
# Model comparison
# train baseline models
baseline_model = baseline(48, 48, 3)
baseline_model.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=['accuracy'])
history1 = baseline_model.fit(LR_patch_train, HR_patch_train, epochs=2, verbose=1,
                    batch_size=4, validation_split=0.2)
# baseline_model.save('./baseline.h5')

# pre-upsampling network
architecture_model = model1(48, 48, 3)
architecture_model.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=['accuracy'])
history2 = architecture_model.fit(LR_patch_train, HR_patch_train, epochs=2, verbose=1,
                    batch_size=4, validation_split=0.2)
# architecture_model.save('./architecture_model.h5')


# change loss function from MSE to defined perceptual loss
perceptual_loss_model = baseline(48, 48, 3)
perceptual_loss_model.compile(optimizer=Adam(lr=1e-4), loss=perceptual_loss, metrics=['accuracy'])
history3 = perceptual_loss_model.fit(LR_patch_train, HR_patch_train, epochs=2, verbose=1,
                    batch_size=4, validation_split=0.2)
# perceptual_loss_model.save('./perceptual_loss_model.h5')


# modify the upsampling layer to subpixel layer
subpixel_model = model2(48, 48, 3)
subpixel_model.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=['accuracy'])
history4 = subpixel_model.fit(LR_patch_train, HR_patch_train, epochs=2, verbose=1,
                    batch_size=4, validation_split=0.2)
# subpixel_model.save('./subpixel_model.h5')


# input patch size = 16*16
model_1616 = baseline(16, 16, 3)
model_1616.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=['accuracy'])
history5 = model_1616.fit(LR_patch_train, HR_patch_train, epochs=2, verbose=1,
                    batch_size=4, validation_split=0.2)
# model_1616.save('./model_1616.h5')


# input patch size = 32*32
model_3232 = baseline(32, 32, 3)
model_3232.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=['accuracy'])
history6 = model_3232.fit(LR_patch_train, HR_patch_train, epochs=2, verbose=1,
                    batch_size=4, validation_split=0.2)
# model_3232.save('./model_3232.h5')


# input6 patch size = 64*64
model_6464 = baseline(64, 64, 3)
model_6464.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=['accuracy'])
history7 = model_6464.fit(LR_patch_train, HR_patch_train, epochs=2, verbose=1,
                    batch_size=4, validation_split=0.2)
# model_6464.save('./model_6464.h5')


# -----------------------------------------------------------------------------
# Compare performance on the first 10 test images
LR_valid_imgs = load_data_from_dir(LR_valid_bicubic_path)
HR_valid_imgs = load_data_from_dir(HR_valid_path)

LR_valid_imgs = LR_valid_imgs[:2]
HR_valid_imgs = HR_valid_imgs[:2]

LR_patch_train = normalize(LR_patch_train)
HR_patch_train = normalize(HR_patch_train)

# change model path to evaluate different models 
model = load_model('./baseline.h5', custom_objects={'tf': tf})

# predict and reconstruct test images
test_num = 2
patch_height = 48
patch_width = 48
stride = 40
up_scale = 2
predicted_HR_list = test_patch(LR_valid_imgs, test_num, patch_height, patch_width, stride, model, up_scale)

# compare with HR images
# calculate PSNR(peak_signal_noise_ratio) and SSIM(structural_similarity) metrics
PSNR_val = []
SSIM_val = []

for i in range(len(predicted_HR_list)):
    PSNR = peak_signal_noise_ratio(HR_valid_imgs[i], predicted_HR_list[i])
    SSIM = structural_similarity(HR_valid_imgs[i], predicted_HR_list[i], multichannel=True)
    PSNR_val.append(PSNR)
    SSIM_val.append(SSIM)

print('PSNR: ', PSNR_val)
print('SSIM: ', SSIM_val)





