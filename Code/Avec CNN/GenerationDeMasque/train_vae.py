import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from vae_model import VAE

def load_images_from_folder(folder, target_size=(256, 256)):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            img = load_img(os.path.join(folder, filename), target_size=target_size, color_mode="grayscale")
            images.append(img_to_array(img) / 255.0)
    return np.array(images)

degraded_images = load_images_from_folder("Dataset_preprocessed/degraded_images")
mask_images = load_images_from_folder("Dataset_preprocessed/masks")

vae = VAE(latent_dim=256)
vae.compile(optimizer="adam", loss="binary_crossentropy")

vae.fit(degraded_images, mask_images, batch_size=32, epochs=10)

vae.save_weights("vae_weights_256x256_segmentation.weights.h5")
