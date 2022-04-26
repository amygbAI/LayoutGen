import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os,cv2
import load_data

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#data_ = load_data.returnBinData()
folder_ = './AUG/'
#folder_ = './doc_images/'
ll_ = os.listdir( folder_ )
annealing_factor = 15
data_ = []
for elem_ctr in range(len(ll_)):
    elem = ll_[ elem_ctr ]
    img = cv2.imread( folder_ + elem, 0 )
    img = cv2.resize( img, (64, 64), cv2.INTER_AREA )
    img = 255.0 - img
    data_.append( np.expand_dims( img, -1 ).astype("float32") / 255 )
    if elem_ctr > 40000: break
    #data_.append( img )

data_ = np.asarray( data_ )
np.random.shuffle( data_ )
#data_ = data_[:40000]
#data_ = np.expand_dims( data_, axis=-1 )
print('TINFOIL HERE', data_.shape)
data_ = tf.convert_to_tensor( data_ , dtype=tf.float32 )
_dataset = tf.data.Dataset.from_tensor_slices( data_ )

dataset = _dataset.shuffle(buffer_size=128)
#dataset = _dataset.map(preprocess_triplets)
image_count = data_.shape[0]
print( '-------------', image_count )
# Let's now split our dataset in train and validation.
train_dataset = dataset.take(int(image_count * 0.95))
val_dataset = dataset.skip(int(image_count * 0.05))

print( len(list(train_dataset.as_numpy_iterator())) )
train_dataset = train_dataset.batch(64, drop_remainder=False)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

val_dataset = val_dataset.batch(64, drop_remainder=False)
val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)


discriminator = keras.Sequential(
    [
        keras.Input(shape=(64, 64, 1)),
        layers.Conv2D(512, kernel_size=8, strides=2, activation="relu", padding="same"),
        layers.Conv2D(256, kernel_size=4, strides=2, activation="relu", padding="same"),
        layers.Conv2D(128, kernel_size=3, strides=2, activation="relu", padding="same"),
        layers.Conv2D(64 , kernel_size=2, strides=1, activation="relu", padding="same"),
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ],
    name="discriminator",
)
discriminator.summary()

'''
as explained in VAE , we shall construct an empty model below and then use "load_weights" in order to initialize it 
PLS ensure that the reconstructed empty model is exactly the same as the one used for training 
'''

latent_dim = 128

latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(8 * 8 * 128, activation="relu")(latent_inputs)
x = layers.Reshape((8, 8, 128))(x)
x = layers.Conv2DTranspose(64, 2, activation="leaky_relu", strides=1, padding="same")(x)
x = layers.Conv2DTranspose(128, 3, activation="leaky_relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(256, 4, activation="leaky_relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(512, 8, activation="leaky_relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
generator = keras.Model(latent_inputs, decoder_outputs, name="decoder_inference")

generator.summary()

gen_step_ctr = 0

class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.step_ctr = 0
        self.offset_loss_weight = 0.1
		'''
		since generator is empty we load the VAE decoder weights here ..if u SKIP this step, the model will still 
		train BUT it wont have a head start on the discriminator leading to poorer results and mode collapse in general 
		'''
        self.generator.load_weights("checkpoint/ckpt")

    def on_epoch_end(self, epoch, logs=None):
        tf.print('GORY->', epoch, output=sys.stdout)

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, real_images):
        # Sample random points in the latent space
        global annealing_factor
        batch_size = tf.shape(real_images)[0]
        print('DUMM->', batch_size)
        self.step_ctr += 1
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Decode them to fake images
        generated_images = self.generator(random_latent_vectors)

        # Combine them with real images
        combined_images = tf.concat([ generated_images, \
                ( real_images ) ], axis=0)

        # Assemble labels discriminating real from fake images
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )
        # Add random noise to the labels - important trick!
        labels += 0.05 * tf.random.normal(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
            #d_loss = self.loss_fn(labels, predictions)*annealing_factor

        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            fakestuff_ = self.generator(random_latent_vectors)
            predictions = self.discriminator( fakestuff_ )
            g_loss = self.loss_fn(misleading_labels, predictions)
            #g_loss += tf.reduce_mean(tf.abs(real_images - fakestuff_))
            #cl_ = tf.reduce_mean(tf.abs(real_images - fakestuff_))

        #grads = tape.gradient( cl_ , self.generator.trainable_weights)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
            "annie": annealing_factor,
        }

class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=3, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.x_off_, self.y_off_ = [] , []

    def on_epoch_end(self, epoch, logs=None):
        import sys
        global annealing_factor
        annealing_factor -= 1
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)
        tf.print( 'AF->', annealing_factor, output_stream=sys.stdout )
        print('DOBU->', generated_images)
		'''
		clubbing results below from 4 samples .. we dont need to stick to this at all ..just helps in adding
		more randomness ..however we will still get the "columnar" effect since there's going to be a neat boundary for every image 
		add more variety in data here 
		'''
        for i in range( 0, self.num_img, 4 ):
            generated_0, generated_1, generated_2, generated_3 = (generated_images[i].numpy())*255,\
                    (generated_images[i+1].numpy())*255, (generated_images[i+2].numpy())*255,\
                    (generated_images[i+3].numpy())*255
            
            upper_ =   np.concatenate(( generated_0, generated_1 ), axis=1)
            lower_ =   np.concatenate(( generated_2, generated_3 ), axis=1)
            fin_img_ = np.concatenate(( upper_, lower_ ), axis=0)

            cv2.imwrite('RES/'+str(epoch)+'_'+str(i)+'_.jpg', fin_img_)

epochs = 10  # In practice, use ~100 epochs

gan = GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss_fn=keras.losses.BinaryCrossentropy(),
)

gan.fit(
    train_dataset, epochs=epochs, \
            callbacks=[GANMonitor(num_img=10*4, latent_dim=latent_dim)]
)

#model.load_weights(checkpoint_filepath)
