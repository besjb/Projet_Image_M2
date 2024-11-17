import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
from tensorflow.keras.utils import array_to_img
from vae_model import VAE

def restore_image(image_path, output_path):
    img = load_img(image_path, target_size=(256, 256), color_mode="grayscale")
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    vae = VAE(latent_dim=256)
    vae.load_weights("best_vae_weights.weights.h5")
    restored, _, _ = vae.predict(img_array)  # Use model to predict
    restored_img = np.squeeze(restored) * 255  # Remove batch dimension and rescale
    restored_img = restored_img.astype(np.uint8)
    restored_img = np.expand_dims(restored_img, axis=-1)  # Add channel dimension
    save_img(output_path, restored_img)

input_folder = "Assets"
output_folder = "Restored_images"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        restore_image(os.path.join(input_folder, filename), os.path.join(output_folder, filename))
