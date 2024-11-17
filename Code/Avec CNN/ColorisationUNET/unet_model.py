from tensorflow.keras import layers, Model

def build_unet(input_shape=(256, 256, 1)):
    """Construit le modèle U-Net."""
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(p1)
    c2 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(p2)
    c3 = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(p3)
    c4 = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(c4)

    # Decoder
    u3 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(c4)
    u3 = layers.concatenate([u3, c3])
    c5 = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(u3)
    c5 = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(c5)

    u2 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c5)
    u2 = layers.concatenate([u2, c2])
    c6 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(u2)
    c6 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(c6)

    u1 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c6)
    u1 = layers.concatenate([u1, c1])
    c7 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(u1)
    c7 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(c7)

    outputs = layers.Conv2D(3, (1, 1), activation="sigmoid")(c7)

    return Model(inputs, outputs)
