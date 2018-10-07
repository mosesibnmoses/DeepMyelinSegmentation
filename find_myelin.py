import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from data import load_test_data

from train import dice_coef_loss, dice_coef, get_unet

weights_path = 'bestmyelin.h5'

img_rows = 1024
img_cols = 1024

def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True, mode="constant")

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p

def load_trained_model():
    model=get_unet()
    model.load_weights(weights_path)

    imgs_test, imgs_id_test = load_test_data()
    imgs_test = preprocess(imgs_test)




    imgs_test = imgs_test.astype('float32')
    mean = np.mean(imgs_test)
    std = np.std(imgs_test)  # std for data normalization

    imgs_test -= mean
    imgs_test /= std

    imgs_mask_test = model.predict(imgs_test, verbose=1,batch_size=1)

    np.save('imgs_mask_test.npy', imgs_mask_test)
    pred_dir = 'preds'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for image, image_id in zip(imgs_mask_test, imgs_id_test):
        image = (image[:, :, 0] * 255.).astype(np.uint8)
        image_resized = resize(image,(1040,1392), mode='constant')
        imsave(os.path.join(pred_dir, str(image_id) + '_pred.png'), image_resized)


if __name__ == "__main__":
    load_trained_model()
