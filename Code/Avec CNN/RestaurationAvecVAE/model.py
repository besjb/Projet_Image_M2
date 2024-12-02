import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Flatten, Reshape, Conv2D, Conv2DTranspose, Lambda

class MultiDomainVAE(Model):
    def __init__(self, latent_dim=128, input_shape=(256, 256, 1), **kwargs):
        super(MultiDomainVAE, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.input_shape = input_shape  # Ajout de cet attribut pour config

        # Encodeurs pour chaque domaine
        self.encoder_X = self.build_encoder(input_shape, name_prefix="X")
        self.encoder_Y = self.build_encoder(input_shape, name_prefix="Y")
        self.encoder_Z = self.build_encoder(input_shape, name_prefix="Z")

        # Décodeurs pour chaque domaine
        self.decoder_X = self.build_decoder(input_shape)
        self.decoder_Y = self.build_decoder(input_shape)
        self.decoder_Z = self.build_decoder(input_shape)

        # Mappings inter-domaines (latents)
        self.mapping_X_to_Y = self.build_mapping(name_prefix="X_to_Y")
        self.mapping_Y_to_Z = self.build_mapping(name_prefix="Y_to_Z")

    def build_encoder(self, input_shape, name_prefix=""):
        inputs = Input(shape=input_shape, name=f"{name_prefix}_input")
        x = Conv2D(32, (3, 3), activation='relu', strides=2, padding='same', name=f"{name_prefix}_conv1")(inputs)
        x = Conv2D(64, (3, 3), activation='relu', strides=2, padding='same', name=f"{name_prefix}_conv2")(x)
        x = Conv2D(128, (3, 3), activation='relu', strides=2, padding='same', name=f"{name_prefix}_conv3")(x)
        x = Flatten(name=f"{name_prefix}_flatten")(x)
        z_mean = Dense(self.latent_dim, name=f"{name_prefix}_z_mean")(x)
        z_log_var = Dense(self.latent_dim, name=f"{name_prefix}_z_log_var")(x)
        z = Lambda(self.reparameterize, name=f"{name_prefix}_z")([z_mean, z_log_var])
        return Model(inputs, [z_mean, z_log_var, z], name=f"{name_prefix}_encoder")

    def build_decoder(self, input_shape):
        """Construction d'un décodeur pour un domaine donné."""
        latent_inputs = Input(shape=(self.latent_dim,))
        x = Dense(32 * 32 * 128, activation='relu')(latent_inputs)
        x = Reshape((32, 32, 128))(x)
        x = Conv2DTranspose(128, (3, 3), activation='relu', strides=2, padding='same')(x)
        x = Conv2DTranspose(64, (3, 3), activation='relu', strides=2, padding='same')(x)
        x = Conv2DTranspose(32, (3, 3), activation='relu', strides=2, padding='same')(x)
        outputs = Conv2DTranspose(input_shape[-1], (3, 3), activation='sigmoid', padding='same')(x)
        return Model(latent_inputs, outputs, name="decoder")

    def build_mapping(self, name_prefix=""):
        """Construction d'un mapping latent entre deux domaines."""
        inputs = Input(shape=(self.latent_dim,), name=f"{name_prefix}_input")
        x = Dense(self.latent_dim, activation='relu', name=f"{name_prefix}_dense1")(inputs)
        x = Dense(self.latent_dim, activation='relu', name=f"{name_prefix}_dense2")(x)
        outputs = Dense(self.latent_dim, name=f"{name_prefix}_output")(x)
        return Model(inputs, outputs, name=f"{name_prefix}_mapping")

    def reparameterize(self, args):
        """Applique le reparamétrage pour échantillonner z."""
        z_mean, z_log_var = args
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def get_config(self):
        """Retourne la configuration pour sérialisation."""
        return {
            "latent_dim": self.latent_dim,
            "input_shape": self.input_shape
        }

    @classmethod
    def from_config(cls, config):
        """Reconstruction du modèle depuis la configuration."""
        return cls(**config)

    def call(self, inputs, training=False):
        """
        Applique l'encodeur, le décodeur et les mappings pour chaque domaine.
        Inputs : Dictionnaire contenant les images des différents domaines.
        """
        # Encodeurs
        z_mean_X, z_log_var_X, z_X = self.encoder_X(inputs['X'])
        z_mean_Y, z_log_var_Y, z_Y = self.encoder_Y(inputs['Y'])
        z_mean_Z, z_log_var_Z, z_Z = self.encoder_Z(inputs['Z'])

        # Mappings latents
        z_Y_mapped = self.mapping_X_to_Y(z_X)  # Mapping X -> Y
        z_Z_mapped = self.mapping_Y_to_Z(z_Y)  # Mapping Y -> Z

        # Décodeurs
        reconstructed_X = self.decoder_X(z_X)
        reconstructed_Y = self.decoder_Y(z_Y)
        reconstructed_Z = self.decoder_Z(z_Z)

        # Si en mode entraînement, retournez également les mappings latents
        if training:
            return {
                'X': reconstructed_X,
                'Y': reconstructed_Y,
                'Z': reconstructed_Z,
                'z_Y_mapped': z_Y_mapped,
                'z_Z_mapped': z_Z_mapped
            }, [z_mean_X, z_log_var_X, z_mean_Y, z_log_var_Y, z_mean_Z, z_log_var_Z]
        else:
            return {
                'X': reconstructed_X,
                'Y': reconstructed_Y,
                'Z': reconstructed_Z,
                'z_Y_mapped': z_Y_mapped,
                'z_Z_mapped': z_Z_mapped
            }
