# extract training patches randomly
# pad testing images to extract overlap patches


import numpy as np
import random
import math
from PIL import Image


# extract training patches randomly
def train_patch(LR_train, HR_train, patch_height, patch_width, patch_num, up_scale=2):
    # Input LR_train and HR_train are list type
    
    img_num = len(LR_train)
    patch_per_img = int(patch_num / img_num)
    
    patches_LR = np.zeros((patch_num, patch_height, patch_width, 3))
    patches_HR = np.zeros((patch_num, patch_height*up_scale, patch_width*up_scale, 3))
    
    k = 0
    for i in range(img_num):
        j = 0
        
        img_height = LR_train[i].shape[0]
        img_width = LR_train[i].shape[1]
        
        while j < patch_per_img:
            x_center = random.randint(0+int(patch_width/2), img_width-int(patch_width/2))
            y_center = random.randint(0+int(patch_height/2), img_height-int(patch_height/2))
            patch_top = y_center+int(patch_height/2)
            patch_bottom = y_center-int(patch_height/2)
            patch_left = x_center-int(patch_width/2)
            patch_right = x_center+int(patch_width/2)
            
            patches_LR[k] = LR_train[i][patch_bottom:patch_top, patch_left:patch_right, :]
            patches_HR[k] = HR_train[i][patch_bottom*up_scale:patch_top*up_scale, 
                                        patch_left*up_scale:patch_right*up_scale, 
                                        :]
            
            j += 1
            k += 1
    
    # change data type from float to uint8 so that the Image.fromarray function can work
    return patches_LR.astype(np.uint8), patches_HR.astype(np.uint8)


# extract overlap patches for testing
def test_patch(LR_test, test_num, patch_height, patch_width, stride, model, up_scale=2):
    # LR_test is list type, should be normalized before the function is called
    
    LR_test = LR_test[:test_num]
    predicted_HR_list = []
    
    for i in range(test_num):
        temp_img = LR_test[i]
        img_height = temp_img.shape[0]
        img_width = temp_img.shape[1]
        
        # calculate how many pixels should be padded
        needed_h = stride - (img_height - patch_height)%stride
        needed_w = stride - (img_width - patch_width)%stride
        new_h = img_height+needed_h
        new_y = img_width+needed_w


        # predicted_HR_prob records the accumulated value of each pixel
        # sum_map records how many times of a pixel has been predicted
        # avg_map = predicted_HR_prob / sum_map
        
        predicted_HR_prob = np.zeros((new_h*up_scale, new_y*up_scale, 3))
        sum_map = np.zeros((new_h*up_scale, new_y*up_scale))
        avg_map = np.zeros((new_h*up_scale, new_y*up_scale, 3))
        
        temp_img_extended = np.zeros((new_h, new_y, 3))
        temp_img_extended[:img_height, :img_width, :] = temp_img
        
        # traverse each image and accumulate the predicted value of each pixel
        predicted_HR_prob, sum_map = predict_patches(temp_img_extended, patch_height, patch_width, up_scale,
                                                     stride, model, predicted_HR_prob, sum_map, start_point='top_left')

        
        for index in range(predicted_HR_prob.shape[2]):
            avg_map[:,:,index] = predicted_HR_prob[:, :, index] / sum_map
        
        # remove padded pixels
        cut_edge_avg_map = avg_map[:img_height*up_scale, :img_width*up_scale, :]
        predicted_HR_list.append(cut_edge_avg_map)
    
    return predicted_HR_list






# traverse each image and accumulate the predicted value of each pixel
def predict_patches(temp_img, patch_height, patch_width, up_scale, stride, model, predicted_HR_prob, sum_map, start_point):
    
    img_height = temp_img.shape[0]
    img_width = temp_img.shape[1]
    
    num_y = int((img_height-patch_height) // stride)+1
    num_x = int((img_width-patch_width) // stride)+1
    
    # choose form four start point to traverse the image
    if start_point == 'top_left':
        start_y = 0
        for j in range(num_y):
            start_x = 0
            for k in range(num_x):
                temp_patch = temp_img[start_y:(start_y+patch_height), start_x:(start_x+patch_width), :]

                # reshape to fit tensor shape
                predicted_patch = model.predict(temp_patch.reshape(1, patch_height, patch_width, 3))
                
                # add values of predicted_patch to the corresponding pixels in predicted_HR_prob
                predicted_HR_prob[start_y*up_scale : (start_y+patch_height)*up_scale, 
                                  start_x*up_scale : (start_x+patch_width)*up_scale,
                                  :] += predicted_patch.reshape(patch_height*up_scale, patch_width*up_scale, 3)
                
                sum_map[start_y*up_scale : (start_y+patch_height)*up_scale,
                        start_x*up_scale : (start_x+patch_width)*up_scale] += 1
                
                start_x += stride
            start_y += stride
            
    elif start_point == 'bottom_right':
        start_y = img_height
        for j in range(num_y):
            start_x = img_width
            for k in range(num_x):
                temp_patch = temp_img[(start_y-patch_height):start_y, (start_x-patch_width):start_x, :]
                predicted_patch = model.predict(temp_patch.reshape(1, patch_height, patch_width, 3))
                
                predicted_HR_prob[(start_y-patch_height)*up_scale : start_y*up_scale, 
                                  (start_x-patch_width)*up_scale : start_x*up_scale,
                                  :] += predicted_patch.reshape(patch_height*up_scale, patch_width*up_scale, 3)
                
                sum_map[(start_y-patch_height)*up_scale : start_y*up_scale,
                        (start_x-patch_width)*up_scale : start_x*up_scale] += 1
                
                start_x -= stride
            start_y -= stride
    
    elif start_point == 'top_right':
        start_y = 0
        for j in range(num_y):
            start_x = img_width
            for k in range(num_x):
                temp_patch = temp_img[start_y : (start_y+patch_height), (start_x-patch_width) : start_x, :]
                predicted_patch = model.predict(temp_patch.reshape(1, patch_height, patch_width, 3))
                
                predicted_HR_prob[start_y*up_scale : (start_y+patch_height)*up_scale, 
                                  (start_x-patch_width)*up_scale : start_x*up_scale,
                                  :] += predicted_patch.reshape(patch_height*up_scale, patch_width*up_scale, 3)
                
                sum_map[start_y*up_scale : (start_y+patch_height)*up_scale,
                        (start_x-patch_width)*up_scale : start_x*up_scale] += 1
                
                start_x -= stride
            start_y += stride
    
    elif start_point == 'bottom_left':
        start_y = img_height
        for j in range(num_y):
            start_x = 0
            for k in range(num_x):
                temp_patch = temp_img[(start_y-patch_height) : start_y, start_x : (start_x+patch_width), :]
                predicted_patch = model.predict(temp_patch.reshape(1, patch_height, patch_width, 3))
                
                predicted_HR_prob[(start_y-patch_height)*up_scale : start_y*up_scale, 
                                  start_x*up_scale : (start_x+patch_width)*up_scale,
                                  :] += predicted_patch.reshape(patch_height*up_scale, patch_width*up_scale, 3)
                
                sum_map[(start_y-patch_height)*up_scale : start_y*up_scale,
                        start_x*up_scale : (start_x+patch_width)*up_scale] += 1
                
                start_x += stride
            start_y -= stride
    

    return predicted_HR_prob, sum_map

