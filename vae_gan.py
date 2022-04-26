import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os, cv2

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

latent_dim = 128

encoder_inputs = keras.Input(shape=(64, 64, 1))
#encoder_inputs = keras.Input(shape=(200, 200, 1))
x = layers.Conv2D(512, 8, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(256, 4, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2D(128, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2D(64, 2, activation="relu", padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(256, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(8 * 8 * 128, activation="relu")(latent_inputs)
x = layers.Reshape((8, 8, 128))(x)
x = layers.Conv2DTranspose(64, 2, activation="leaky_relu", strides=1, padding="same")(x)
x = layers.Conv2DTranspose(128, 3, activation="leaky_relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(256, 4, activation="leaky_relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(512, 8, activation="leaky_relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
#decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

'''
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(8 * 8 * 128, activation="relu")(latent_inputs)
x = layers.Reshape((8, 8, 128))(x)
x = layers.Conv2DTranspose(64, 2, activation="leaky_relu", strides=1, padding="same")(x)
x = layers.Conv2DTranspose(128, 3, activation="leaky_relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(256, 4, activation="leaky_relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(512, 8, activation="leaky_relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
#decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
decoder_empty = keras.Model(latent_inputs, decoder_outputs, name="decoder_inference")
'''

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        #grads = tape.gradient( reconstruction_loss, self.trainable_weights)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

'''
the section below is a generic way to load data ..the folder SHOULD contain all the images that you want to approximate
using VAE ..doesnt matter what kind BUT for the initial set of experiments i used a contoured image with black background
will store a few in the git folder for samples ..please feel free to use any kind of image
'''
folder_ = './AUG/'
ll_ = os.listdir( folder_ )
data_ = []
for elem_ctr in range(len(ll_)):
    elem = ll_[ elem_ctr ]
    img = cv2.imread( folder_ + elem )
    img = cv2.imread( folder_ + elem, 0 )
    img = cv2.resize( img, (64, 64), cv2.INTER_AREA )
    img = 255.0 - img
    '''
    if elem_ctr == 0: 
        cv2.imwrite( 'tt1.jpg', img1 )
        cv2.imwrite( 'tt2.jpg', img )
        exit()
    '''
    #img = cv2.resize( img, (200, 200), cv2.INTER_AREA )
    #data_.append( img.astype("float32") / 255 )
    data_.append( np.expand_dims( img, -1 ).astype("float32") / 255 )
    if elem_ctr > 50000: break

data_ = np.asarray( data_ )
np.random.shuffle( data_ )


vae = VAE(encoder, decoder)
vae.compile( optimizer=keras.optimizers.Adam() )
vae.fit( data_ , epochs=30, batch_size=128)

decoder.save( 'VAE_PASS1_DECODER' )

'''
whenever we want to use a model for TRANSFER LEARNING we simply
a) save the weights
b) create any empty model wherever we want to use items
c) inititalize the empty model with the weights saved here 
'''
decoder.save_weights("checkpoint/ckpt")

import matplotlib.pyplot as plt


def plot_latent_space(vae, n=30, figsize=15):
    # display a n*n 2D manifold of digits
    digit_size = 64
    scale = 1.0
    figure = np.ones((digit_size * n, digit_size * n, 3))*255.0
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    decoder_ = keras.models.load_model( 'VAE_PASS1_DECODER' )
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.expand_dims( np.random.normal( 0, 1, latent_dim ), 0 )
            #z_sample = np.array([[xi, yi]])
            print( z_sample.shape )
            x_decoded = decoder_.predict(z_sample)
            #x_decoded = vae.decoder.predict(z_sample)
            
            digit = x_decoded[0].reshape(digit_size, digit_size, 1)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
                :
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.savefig('Sample1.jpg')


plot_latent_space(vae, n=5)


