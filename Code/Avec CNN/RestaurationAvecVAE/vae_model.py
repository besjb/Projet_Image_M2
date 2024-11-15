import tensorflow as tf
from tensorflow.keras import layers, Model

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
            layers.Dense(latent_dim + latent_dim)  # mean + logvar
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

    def call(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar
