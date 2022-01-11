# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 10:02:17 2019

@author: Artur Yakimovich, PhD (c) 2019. UCL
"""
import time
import os
import unet7
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.callbacks import TensorBoard
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import resize
#from skimage.util import crop
from skimage.io import imsave, imread

def checkDir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def read_imgs(dir, img_rows, img_cols):

    images = [f for f in os.listdir(dir) if f.endswith('.png')]

    imgs = np.ndarray((len(images), img_rows, img_cols, 1), dtype=np.float)
    
    for idx, img in enumerate(images):
        #print(idx)
        img = imread(os.path.join(dir, img), as_gray=True)
        img = np.expand_dims(img, axis=-1)
        imgs[idx] = img
    return imgs

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

img_cols = 900
img_rows = 900
epochs = 1001
batch_size = 4
smooth = 1.
target_width = 480
target_height = 480
start_lr = 5e-8

upscale_factor_width = img_cols / target_width
upscale_factor_height = img_cols / target_height

runningTime = time.strftime('%b-%d-%Y_%H-%M')
model_dir = 'R:\\Data\\180321_DeepLongevity\\unet\\'
log_dir = os.path.join(model_dir,'logs',"{}".format(runningTime))
checkDir(log_dir)

training_img_dir = os.path.join(model_dir,'data','train','img')
training_mask_dir = os.path.join(model_dir,'data','train','mask')
test_img_dir = os.path.join(model_dir,'data','test','img')
test_mask_dir = os.path.join(model_dir,'data','test','mask')


# load training images

training_imgs = read_imgs(training_img_dir, img_rows, img_cols)
plt.figure()
plt.title('Example image')
plt.imshow(np.squeeze(training_imgs[0]))

training_imgs = training_imgs.astype('float32')
training_imgs = training_imgs / training_imgs.max()


training_masks = read_imgs(training_mask_dir, img_rows, img_cols)
plt.figure()
plt.title('Example mask')
plt.imshow(np.squeeze(training_masks[0]))

training_masks = training_masks.astype('float32')
training_masks = training_masks > 0.5


# Load test images

test_imgs = read_imgs(test_img_dir, img_rows, img_cols)
plt.figure()
plt.title('Example image')
plt.imshow(np.squeeze(test_imgs[0]))

test_imgs = test_imgs.astype('float32')
test_imgs = test_imgs / test_imgs.max()


test_masks = read_imgs(test_mask_dir, img_rows, img_cols)
plt.figure()
plt.title('Example mask')
plt.imshow(np.squeeze(test_masks[0]))

test_masks = test_masks.astype('float32')
test_masks = test_masks > 0.5

# resize images to the target size to fit GPU

n_imgs = training_imgs.shape[0]
training_imgs = resize(training_imgs, (n_imgs, target_width, target_height, 1))
training_masks = resize(training_masks, (n_imgs, target_width, target_height, 1))


n_imgs = test_imgs.shape[0]
test_imgs = resize(test_imgs, (n_imgs, target_width, target_height, 1))
test_masks = resize(test_masks, (n_imgs, target_width, target_height, 1))


# pad images to fit the network
'''
pad_by = 28
training_imgs = np.pad(training_imgs, [(0, 0), (0, pad_by), (0, pad_by), (0, 0)],
                                       mode='constant', constant_values=0)
training_masks = np.pad(training_masks, [(0, 0), (0, pad_by), (0, pad_by), (0, 0)],
                                       mode='constant', constant_values=0)

test_imgs = np.pad(test_imgs, [(0, 0), (0, pad_by), (0, pad_by), (0, 0)],
                                       mode='constant', constant_values=0)
test_masks = np.pad(test_masks, [(0, 0), (0, pad_by), (0, pad_by), (0, 0)],
                                       mode='constant', constant_values=0)

img_cols = 928
img_rows = 928
'''

# Data augmentation

data_gen_args = dict(featurewise_center=False,
                     featurewise_std_normalization=False,
                     rotation_range=50.,
                     width_shift_range=0.01,
                     height_shift_range=0.01,
                     zoom_range=0.1,
                     fill_mode='constant',
                     cval=0)

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

seed = 123
image_datagen.fit(training_imgs, augment=True, seed=seed)
mask_datagen.fit(training_masks, augment=True, seed=seed)

image_generator = image_datagen.flow(training_imgs, batch_size=batch_size, seed=seed)
mask_generator = mask_datagen.flow(training_masks, batch_size=batch_size, seed=seed)

gen1 = zip(image_generator, mask_generator)


# Training


tensorboard = TensorBoard(log_dir=os.path.join(model_dir,"{}".format(runningTime)))
callbacksList = []

model = unet7.get_unet(target_width, target_height)

#model.compile(optimizer=Adam(lr=5e-5), loss=unet7.dice_coef_loss, metrics=[unet7.dice_coef])
model.compile(optimizer=Adam(lr=start_lr), loss=unet7.dice_coef_loss, metrics=[unet7.dice_coef])

model.summary()
print("start tesorboard, cmd: tensorboard --logdir=\""+os.path.join(log_dir,"{}".format(runningTime)+"\""))

loss_history_list = []
dice_history_list = []
epoch_list = []
for e in range(epochs):

    batches = 0
    for x_batch, y_batch in gen1:
        y_batch = y_batch > 0.5
        history_callback = model.fit(x_batch, y_batch, epochs=1, verbose=0, callbacks=callbacksList)
        batches += 1
        if batches >= len(training_imgs) / batch_size:
            break
    loss_history_list.append(history_callback.history["loss"])
    dice_history_list.append(history_callback.history["dice_coef"])
    epoch_list.append(e)
    print('Epoch {}, dice_coef {}'.format(e,history_callback.history["dice_coef"]))
    if e % 50 == 0:

        #print('Epoch', e)

        # Show a single case

        

        pred1 = model.predict(test_imgs)

        plt.figure()
        plt.subplot(1, 3, 1)
        plt.title('Example')
        plt.imshow(np.squeeze(test_imgs[1]))
        plt.subplot(1, 3, 2)
        plt.title('Model Epoch {}'.format(e))
        plt.imshow(np.squeeze(pred1[1]))
        plt.subplot(1, 3, 3)
        plt.title('Ground truth')
        plt.imshow(np.squeeze(test_masks[1]))
        plt.savefig(os.path.join(log_dir,'ground_truth_e{}.pdf'.format(e)), bbox_inches='tight', dpi=300)
        plt.show()

        model.save_weights(model_dir + "modelaug2-{}.h5".format(e))
        
train_history = pd.DataFrame({"epoch":epoch_list, "loss":loss_history_list, "dice_coef":dice_history_list})
train_history.to_csv(os.path.join(log_dir,"history.csv"))