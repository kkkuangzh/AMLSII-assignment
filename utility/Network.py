import tensorflow as tf
from keras.layers import Lambda
from keras.models import Model, Sequential
from keras.layers import PReLU, Input, Conv2D, UpSampling2D, add


# define subpixel layer for up-sampling
def pixelshuffler(input_shape, batch_size, scale=2):
    def subpixel_shape(input_shape=input_shape, batch_size=batch_size):
        dim = [batch_size,
               input_shape[1] * scale,
               input_shape[2] * scale,
               int(input_shape[3]/ (scale ** 2))]
        output_shape = tuple(dim)

        return output_shape

    def pixelshuffle_upscale(x):
        return tf.nn.depth_to_space(input=x, block_size=scale)

    return Lambda(function=pixelshuffle_upscale, output_shape=subpixel_shape)


# residual block
def res_block(inputs):
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    x = PReLU(shared_axes=[1, 2])(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    return add([x, inputs])


def baseline(patch_height, patch_width, channel, upscale=2):
    # conv and then upsample

    inputs = Input(shape=(patch_height, patch_width, channel))
    x_init = Conv2D(filters=64, kernel_size=(9, 9), strides=(1, 1), padding='same')(inputs)
    x = PReLU(shared_axes=[1, 2])(x_init)

    # residual_block
    for i in range(8):
        x = res_block(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = add([x, x_init])

    # up_block
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = UpSampling2D(size=(2, 2))(x)  # size:upsampling factor
    x = PReLU(shared_axes=[1, 2])(x)

    # output_block
    output = Conv2D(filters=3, kernel_size=(9, 9), strides=(1, 1), padding='same')(x)
    output = Conv2D(3, (1, 1), activation='sigmoid', padding='same')(output)

    model = Model(inputs=inputs, outputs=output)

    return model


# modify the location of upsampling layer
# define pre-upsampling network architecture
def model1(patch_height, patch_width, channel, upscale=2):
    # upsample and then conv

    inputs = Input(shape=(patch_height, patch_width, channel))
    x_init = Conv2D(filters=64, kernel_size=(9, 9), strides=(1, 1), padding='same')(inputs)
    x = PReLU(shared_axes=[1, 2])(x_init)

    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = add([x, x_init])

    # up_block
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = PReLU(shared_axes=[1, 2])(x)

    # residual_block
    for i in range(8):
        x = res_block(x)

    # output_block
    output = Conv2D(filters=3, kernel_size=(9, 9), strides=(1, 1), padding='same')(x)
    output = Conv2D(3, (1, 1), activation='sigmoid', padding='same')(output)

    model = Model(inputs=inputs, outputs=output)

    return model


# modify upsampling layer to subpixel layer
def model2(patch_height, patch_width, channel, upscale=2):
    # conv and then upsample

    inputs = Input(shape=(patch_height, patch_width, channel))
    x_init = Conv2D(filters=64, kernel_size=(9, 9), strides=(1, 1), padding='same')(inputs)
    x = PReLU(shared_axes=[1, 2])(x_init)

    # residual_block
    for i in range(8):
        x = res_block(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = add([x, x_init])

    # up_block
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = pixelshuffler(input_shape=(48, 48, 3), batch_size=4, scale=upscale)(x)
    x = PReLU(shared_axes=[1, 2])(x)

    # output_block
    output = Conv2D(filters=3, kernel_size=(9, 9), strides=(1, 1), padding='same')(x)
    output = Conv2D(3, (1, 1), activation='sigmoid', padding='same')(output)

    model = Model(inputs=inputs, outputs=output)

    return model


# after comparing the modifications mentioned above, the final network architecture
def final_model(patch_height, patch_width, channel, upscale=2):
    # conv and then upsample

    inputs = Input(shape=(patch_height, patch_width, channel))
    x_init = Conv2D(filters=64, kernel_size=(9, 9), strides=(1, 1), padding='same')(inputs)
    x = PReLU(shared_axes=[1, 2])(x_init)

    # residual block
    for i in range(8):
        x = res_block(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = add([x, x_init])

    # sub-pixel up_block
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = pixelshuffler(input_shape=(96, 96, 3), batch_size=4, scale=upscale)(x)
    x = PReLU(shared_axes=[1, 2])(x)

    # output_block
    output = Conv2D(filters=3, kernel_size=(9, 9), strides=(1, 1), padding='same')(x)
    output = Conv2D(3, (1, 1), activation='sigmoid', padding='same')(output)

    model = Model(inputs=inputs, outputs=output)

    return model



