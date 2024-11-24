import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Dense, Flatten, Reshape, Conv2D, Conv2DTranspose, Lambda,
    Concatenate, UpSampling2D
)


class MultiDomainVAE(Model):
    def __init__(self, latent_dim=128, input_shape=(256, 256, 1), **kwargs):
        super(MultiDomainVAE, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.input_shape_model = input_shape  # Utilisé pour sérialisation

        # Encodeurs pour chaque domaine
        self.encoder_X = self.build_encoder(input_shape, name_prefix="X")
        self.encoder_Y = self.build_encoder(input_shape, name_prefix="Y")
        self.encoder_Z = self.build_encoder(input_shape, name_prefix="Z")

        # Décodeurs pour chaque domaine
        self.decoder_X = self.build_decoder(input_shape, name_prefix="X")
        self.decoder_Y = self.build_decoder(input_shape, name_prefix="Y")
        self.decoder_Z = self.build_decoder(input_shape, name_prefix="Z")

    def get_config(self):
        """Retourne la configuration pour sérialisation."""
        config = super(MultiDomainVAE, self).get_config()
        config.update({
            "latent_dim": self.latent_dim,
            "input_shape_model": self.input_shape_model,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Reconstruction du modèle depuis la configuration."""
        return cls(
            latent_dim=config["latent_dim"],
            input_shape=config["input_shape_model"],
        )

    def build_encoder(self, input_shape, name_prefix=""):
        inputs = Input(shape=input_shape, name=f"{name_prefix}_input")

        # Couches descendantes (down-sampling)
        x1 = Conv2D(32, (3, 3), activation='relu', strides=2, padding='same', name=f"{name_prefix}_conv1")(inputs)
        x2 = Conv2D(64, (3, 3), activation='relu', strides=2, padding='same', name=f"{name_prefix}_conv2")(x1)
        x3 = Conv2D(128, (3, 3), activation='relu', strides=2, padding='same', name=f"{name_prefix}_conv3")(x2)

        # Goulot d'étranglement (latent space)
        x_flat = Flatten(name=f"{name_prefix}_flatten")(x3)
        z_mean = Dense(self.latent_dim, name=f"{name_prefix}_z_mean")(x_flat)
        z_log_var = Dense(self.latent_dim, name=f"{name_prefix}_z_log_var")(x_flat)
        z = Lambda(self.reparameterize, name=f"{name_prefix}_z")([z_mean, z_log_var])

        return Model(inputs, [z_mean, z_log_var, z, x1, x2, x3], name=f"{name_prefix}_encoder")

    def build_decoder(self, input_shape, name_prefix=""):
        latent_inputs = Input(shape=(self.latent_dim,), name=f"{name_prefix}_latent_input")
        x = Dense(16 * 16 * 128, activation='relu', name=f"{name_prefix}_dense")(latent_inputs)
        x = Reshape((16, 16, 128), name=f"{name_prefix}_reshape")(x)

        skip1 = Input(shape=(128, 128, 32), name=f"{name_prefix}_skip1")
        skip2 = Input(shape=(64, 64, 64), name=f"{name_prefix}_skip2")
        skip3 = Input(shape=(32, 32, 128), name=f"{name_prefix}_skip3")

        # Reconstruction avec connexions de saut
        x = Conv2DTranspose(128, (3, 3), activation='relu', strides=2, padding='same', name=f"{name_prefix}_deconv1")(x)
        x = Concatenate(name=f"{name_prefix}_concat1")([x, skip3])  # Résolution (32, 32)

        x = Conv2DTranspose(64, (3, 3), activation='relu', strides=2, padding='same', name=f"{name_prefix}_deconv2")(x)
        x = Concatenate(name=f"{name_prefix}_concat2")([x, skip2])  # Résolution (64, 64)

        x = Conv2DTranspose(32, (3, 3), activation='relu', strides=2, padding='same', name=f"{name_prefix}_deconv3")(x)
        x = Concatenate(name=f"{name_prefix}_concat3")([x, skip1])  # Résolution finale (128, 128)

        outputs = Conv2DTranspose(input_shape[-1], (3, 3), activation='sigmoid', padding='same', name=f"{name_prefix}_output")(x)

        return Model([latent_inputs, skip1, skip2, skip3], outputs, name=f"{name_prefix}_decoder")

    def reparameterize(self, args):
        """Applique le reparamétrage pour échantillonner z."""
        z_mean, z_log_var = args
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def call(self, inputs, training=False):
        try:
            # Encodeurs
            z_mean_X, z_log_var_X, z_X, skip1_X, skip2_X, skip3_X = self.encoder_X(inputs['X'])
            z_mean_Y, z_log_var_Y, z_Y, skip1_Y, skip2_Y, skip3_Y = self.encoder_Y(inputs['Y'])
            z_mean_Z, z_log_var_Z, z_Z, skip1_Z, skip2_Z, skip3_Z = self.encoder_Z(inputs['Z'])

            print("Encoder outputs validated")

            # Décodeurs avec connexions de saut
            reconstructed_X = self.decoder_X([z_X, skip1_X, skip2_X, skip3_X])
            reconstructed_Y = self.decoder_Y([z_Y, skip1_Y, skip2_Y, skip3_Y])
            reconstructed_Z = self.decoder_Z([z_Z, skip1_Z, skip2_Z, skip3_Z])

            print("Decoder outputs validated")

            # Redimensionner si nécessaire
            reconstructed_X = tf.image.resize(reconstructed_X, [self.input_shape_model[0], self.input_shape_model[1]])
            reconstructed_Y = tf.image.resize(reconstructed_Y, [self.input_shape_model[0], self.input_shape_model[1]])
            reconstructed_Z = tf.image.resize(reconstructed_Z, [self.input_shape_model[0], self.input_shape_model[1]])

            print("Reconstructed shapes after resizing:", reconstructed_X.shape, reconstructed_Y.shape, reconstructed_Z.shape)

            if training:
                return {
                    'X': reconstructed_X,
                    'Y': reconstructed_Y,
                    'Z': reconstructed_Z
                }, {
                    'z_mean_X': z_mean_X, 'z_log_var_X': z_log_var_X,
                    'z_mean_Y': z_mean_Y, 'z_log_var_Y': z_log_var_Y,
                    'z_mean_Z': z_mean_Z, 'z_log_var_Z': z_log_var_Z
                }
            else:
                return {
                    'X': reconstructed_X,
                    'Y': reconstructed_Y,
                    'Z': reconstructed_Z
                }

        except Exception as e:
            print("Error in call method:", str(e))
            raise

    def compute_loss(self, inputs, outputs, z_means, z_log_vars):
        """Calcul des pertes de reconstruction et divergence KL."""
        reconstruction_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(inputs, outputs)
        )
        reconstruction_loss *= self.input_shape_model[0] * self.input_shape_model[1]
        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_vars - tf.square(z_means) - tf.exp(z_log_vars)
        )
        return reconstruction_loss + kl_loss
