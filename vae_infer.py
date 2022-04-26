import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os, cv2

import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def plot_latent_space():
    # display a n*n 2D manifold of digits
    digit_size = 64
    latent_dim = 128
    scale = 1.0
    decoder_ = keras.models.load_model( 'VAE_PASS1_DECODER' )

    for ctr in range(100):
            z_sample = np.expand_dims( np.random.normal( 0, 1, latent_dim ), 0 )
            x_decoded_0 = decoder_.predict(z_sample)
            z_sample = np.expand_dims( np.random.normal( 0, 1, latent_dim ), 0 )
            x_decoded_1 = decoder_.predict(z_sample)
            z_sample = np.expand_dims( np.random.normal( 0, 1, latent_dim ), 0 )
            x_decoded_2 = decoder_.predict(z_sample)
            z_sample = np.expand_dims( np.random.normal( 0, 1, latent_dim ), 0 )
            x_decoded_3 = decoder_.predict(z_sample)
            
            digit = (( x_decoded_0[0].reshape(digit_size, digit_size, 1) ))*255
            digit2 = (( x_decoded_1[0].reshape(digit_size, digit_size, 1) ))*255
            digit3 = (( x_decoded_2[0].reshape(digit_size, digit_size, 1) ))*255
            digit4 = (( x_decoded_3[0].reshape(digit_size, digit_size, 1) ))*255

            upper_ =   np.concatenate(( digit, digit2 ), axis=1)
            lower_ =   np.concatenate(( digit3, digit4 ), axis=1)
            fin_img_ = np.concatenate(( upper_, lower_ ), axis=0)

            cv2.imwrite('VAE_RES/_'+str(ctr)+'.jpg', fin_img_ )

plot_latent_space()


