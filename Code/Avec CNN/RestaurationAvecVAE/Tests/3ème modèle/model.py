import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Reshape, Conv2D, Conv2DTranspose, Flatten, Lambda, BatchNormalization, Activation, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy

# Bloc ResNet pour les encodeurs/décodeurs
def resnet_block(input_tensor, filters, kernel_size=3, stride=1):
    """
    Implémente un bloc ResNet avec un chemin principal et un chemin de saut.
    """
    x = Conv2D(filters, kernel_size, strides=stride, padding='same', kernel_initializer='he_normal')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size, strides=1, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    shortcut = Conv2D(filters, kernel_size=1, strides=stride, padding='same', kernel_initializer='he_normal')(input_tensor)
    shortcut = BatchNormalization()(shortcut)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

# Classe MultiDomainVAE
class MultiDomainVAE(Model):
    def __init__(self, latent_dim=256, **kwargs):
        super(MultiDomainVAE, self).__init__(**kwargs)
        self.latent_dim = latent_dim

        # Encodeurs et décodeurs pour chaque domaine
        self.encoder_X = self.create_encoder()
        self.encoder_R = self.create_encoder()
        self.encoder_Y = self.create_encoder()
        self.decoder_X = self.create_decoder()
        self.decoder_R = self.create_decoder()
        self.decoder_Y = self.create_decoder()

    def create_encoder(self):
        """
        Crée un encodeur avec des blocs ResNet.
        """
        inputs = Input(shape=(256, 256, 3))
        x = Conv2D(32, (3, 3), strides=1, padding='same', activation='relu')(inputs)
        x = resnet_block(x, 32)
        x = Conv2D(64, (3, 3), strides=2, padding='same', activation='relu')(x)
        x = resnet_block(x, 64)
        x = Conv2D(128, (3, 3), strides=2, padding='same', activation='relu')(x)
        x = resnet_block(x, 128)
        x = Conv2D(256, (3, 3), strides=2, padding='same', activation='relu')(x)
        x = resnet_block(x, 256)
        x = Flatten()(x)

        z_mean = Dense(self.latent_dim)(x)
        z_log_var = Dense(self.latent_dim)(x)
        z = Lambda(lambda args: args[0] + tf.exp(args[1] / 2) * tf.random.normal(shape=(tf.shape(args[0])[0], self.latent_dim)))([z_mean, z_log_var])

        return Model(inputs, [z_mean, z_log_var, z], name="encoder")

    def create_decoder(self):
        """
        Crée un décodeur avec des blocs ResNet.
        """
        latent_inputs = Input(shape=(self.latent_dim,))
        x = Dense(32 * 32 * 256, activation='relu')(latent_inputs)
        x = Reshape((32, 32, 256))(x)
        x = resnet_block(x, 256)
        x = Conv2DTranspose(128, (3, 3), strides=2, padding='same', activation='relu')(x)
        x = resnet_block(x, 128)
        x = Conv2DTranspose(64, (3, 3), strides=2, padding='same', activation='relu')(x)
        x = resnet_block(x, 64)
        x = Conv2DTranspose(32, (3, 3), strides=2, padding='same', activation='relu')(x)
        x = resnet_block(x, 32)
        outputs = Conv2DTranspose(3, (3, 3), strides=1, padding='same', activation='sigmoid')(x)

        return Model(latent_inputs, outputs, name="decoder")


    def call(self, inputs, training=False):
        """
        Mode entraînement (dictionnaire) ou inférence (entrée unique).
        """
        if training:
            # Encodeurs
            z_mean_X, z_log_var_X, z_X = self.encoder_X(inputs['X'])
            z_mean_R, z_log_var_R, z_R = self.encoder_R(inputs['R'])
            z_mean_Y, z_log_var_Y, z_Y = self.encoder_Y(inputs['Y'])

            # Décodeurs
            reconstructed_X = self.decoder_X(z_X)
            reconstructed_R = self.decoder_R(z_R)
            reconstructed_Y = self.decoder_Y(z_Y)

            return {
                'X': reconstructed_X,
                'R': reconstructed_R,
                'Y': reconstructed_Y
            }
        else:
            z_mean, z_log_var, z = self.encoder_X(inputs)
            reconstructed = self.decoder_X(z)
            return reconstructed
