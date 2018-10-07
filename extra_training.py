from __future__ import print_function
import time

import sys
import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K

from train import dice_coef_loss, dice_coef, get_unet
from data import load_train_data, load_test_data

os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #supresses TF warnings
K.set_image_data_format('channels_last')  # TF dimension ordering in this code

img_rows = 1024
img_cols = 1024

smooth = 1.

def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True, mode="constant")
    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p

def train_more(model_in,model_out,num_epochs):
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train, imgs_mask_train = load_train_data()

    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)
    std = np.std(imgs_train)

    imgs_train -= mean
    print(mean)
    print(std)
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scales masks to [0, 1]

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
        model=load_model(model_in, custom_objects={'dice_coef_loss': dice_coef_loss,'dice_coef':dice_coef})
    model_checkpoint = ModelCheckpoint('checkpoint.h5', monitor='val_loss', save_best_only=True)
    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    start=time.time()
    model.fit(imgs_train, imgs_mask_train, batch_size=1, epochs=num_epochs, verbose=1, shuffle=True,
              validation_split=0.2,
              callbacks=[model_checkpoint])
    end=time.time()
    print('time elapsed:',end-start,'seconds')
    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    imgs_test, imgs_id_test = load_test_data()
    imgs_test = preprocess(imgs_test)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('weights.h5')

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_mask_test = model.predict(imgs_test, verbose=1,batch_size=1)
    model.save(model_out)
    np.save('imgs_mask_test.npy', imgs_mask_test)

    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)
    pred_dir = 'preds'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for image, image_id in zip(imgs_mask_test, imgs_id_test):
        image = (image[:, :, 0] * 255.).astype(np.uint8)
        image = resize(image,(1040,1392),mode='constant')
        imsave(os.path.join(pred_dir, str(image_id) + '_pred.png'), image)

if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise ValueError('USAGE: use 3 arguments: model_in, model_out, num_epochs')
    model_in = sys.argv[1]
    model_out= sys.argv[2]
    num_epochs = int(sys.argv[3])
    train_more(model_in,model_out,num_epochs)
