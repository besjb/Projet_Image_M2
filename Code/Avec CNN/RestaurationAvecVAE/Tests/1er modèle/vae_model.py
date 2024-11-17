import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def vae_loss(x, x_decoded_mean, z_mean, z_log_var):
    # Perte de reconstruction
    reconstruction_loss = tf.reduce_mean(
        tf.reduce_sum(tf.keras.losses.binary_crossentropy(x, x_decoded_mean), axis=(1, 2))
    )
    # Divergence KL
    kl_loss = -0.5 * tf.reduce_mean(
        tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
    )
    return reconstruction_loss + kl_loss


class VAE(Model):
    def __init__(self, latent_dim=256):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(256, 256, 1)),
            layers.Conv2D(32, 4, strides=2, padding='same', activation='relu'),
            layers.Conv2D(64, 4, strides=2, padding='same', activation='relu'),
            layers.Conv2D(128, 4, strides=2, padding='same', activation='relu'),
            layers.Conv2D(256, 4, strides=2, padding='same', activation='relu'),
            layers.Flatten(),
            layers.Dense(latent_dim * 2)  # output both mean and logvar, doubled size
        ])

        # Decoder
        self.decoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(latent_dim,)),
            layers.Dense(16 * 16 * 256, activation='relu'),
            layers.Reshape((16, 16, 256)),
            layers.Conv2DTranspose(128, 4, strides=2, padding='same', activation='relu'),
            layers.Conv2DTranspose(64, 4, strides=2, padding='same', activation='relu'),
            layers.Conv2DTranspose(32, 4, strides=2, padding='same', activation='relu'),
            layers.Conv2DTranspose(1, 4, strides=2, padding='same', activation='sigmoid')
        ])

    def encode(self, x):
        mean_logvar = self.encoder(x)
        mean, logvar = tf.split(mean_logvar, num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        epsilon = tf.random.normal(shape=tf.shape(mean))
        return mean + tf.exp(0.5 * logvar) * epsilon

    def decode(self, z):
        return self.decoder(z)

    def call(self, inputs):
        z_mean, z_log_var = self.encode(inputs)
        z = self.reparameterize(z_mean, z_log_var)
        x_decoded_mean = self.decode(z)
        return x_decoded_mean, z_mean, z_log_var


    def train_step(self, data):
        # Unpack data if it's in the form (input, target)
        if isinstance(data, tuple):
            inputs, targets = data
        else:
            inputs = data
            targets = data  # Assuming autoencoder structure where input is target

        with tf.GradientTape() as tape:
            x_decoded_mean, z_mean, z_log_var = self(inputs, training=True)
            loss = vae_loss(targets, x_decoded_mean, z_mean, z_log_var)
        
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {'loss': loss}

