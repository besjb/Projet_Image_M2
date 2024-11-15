import os
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from unet_model import build_unet
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Charger les données
def load_dataset(folder, target_size=(256, 256)):
    gray_images, color_images = [], []
    for filename in os.listdir(folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_color = load_img(os.path.join(folder, filename), target_size=target_size)
            img_gray = load_img(os.path.join(folder, filename), target_size=target_size, color_mode="grayscale")

            gray_images.append(img_to_array(img_gray) / 255.0)
            color_images.append(img_to_array(img_color) / 255.0)

    return np.array(gray_images), np.array(color_images)

gray_images, color_images = load_dataset("Dataset_preprocessed")

# Construire et compiler le modèle
unet_model = build_unet(input_shape=(256, 256, 1))
unet_model.compile(optimizer=Adam(learning_rate=1e-3), loss=MeanSquaredError())

# Entraîner le modèle
unet_model.fit(gray_images, color_images, batch_size=64, epochs=1, validation_split=0.2)

# Sauvegarder le modèle
os.makedirs("models", exist_ok=True)
unet_model.save("models/unet_colorization.h5")
