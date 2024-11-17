import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Reshape, Conv2D, Conv2DTranspose, Flatten, Lambda, BatchNormalization, Activation


class MultiDomainVAE(Model):
    def __init__(self, latent_dim=256, **kwargs):
        super(MultiDomainVAE, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.encoder_X = self.create_encoder()
        self.encoder_R = self.create_encoder()
        self.encoder_Y = self.create_encoder()
        self.decoder_X = self.create_decoder()
        self.decoder_R = self.create_decoder()
        self.decoder_Y = self.create_decoder()

    def create_encoder(self):
        """Create encoder with RGB input."""
        inputs = Input(shape=(256, 256, 3))
        x = Conv2D(32, (3, 3), activation='relu', padding='same', strides=2)(inputs)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', strides=2)(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', strides=2)(x)
        x = Flatten()(x)
        z_mean = Dense(self.latent_dim)(x)
        z_log_var = Dense(self.latent_dim)(x)
        z = Lambda(lambda args: args[0] + tf.exp(args[1] / 2) * tf.random.normal(tf.shape(args[0])))([z_mean, z_log_var])
        return Model(inputs, [z_mean, z_log_var, z], name="encoder")

    def create_decoder(self):
        """Create decoder with RGB output."""
        latent_inputs = Input(shape=(self.latent_dim,))
        x = Dense(16 * 16 * 256, activation='relu')(latent_inputs)
        x = Reshape((16, 16, 256))(x)
        x = Conv2DTranspose(128, (3, 3), activation='relu', padding='same', strides=2)(x)
        x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same', strides=2)(x)
        x = Conv2DTranspose(32, (3, 3), activation='relu', padding='same', strides=2)(x)
        x = Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same', strides=2)(x)
        return Model(latent_inputs, x, name="decoder")

    def call(self, inputs, training=False):
        if training:
            z_mean_X, z_log_var_X, z_X = self.encoder_X(inputs['X'])
            z_mean_R, z_log_var_R, z_R = self.encoder_R(inputs['R'])
            z_mean_Y, z_log_var_Y, z_Y = self.encoder_Y(inputs['Y'])

            reconstructed_X = self.decoder_X(z_X)
            reconstructed_R = self.decoder_R(z_R)
            reconstructed_Y = self.decoder_Y(z_Y)

            return {'X': reconstructed_X, 'R': reconstructed_R, 'Y': reconstructed_Y}
        else:
            z_mean, z_log_var, z = self.encoder_X(inputs)
            reconstructed = self.decoder_X(z)
            return reconstructed
