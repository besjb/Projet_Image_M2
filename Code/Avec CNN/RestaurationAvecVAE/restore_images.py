import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
from vae_model import VAE

vae = VAE(latent_dim=256)
vae.load_weights("vae_weights_256x256.weights.h5")

def restore_image(image_path, output_path):
    img = load_img(image_path, target_size=(256, 256), color_mode="grayscale")
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    restored, _, _ = vae(img_array)
    restored_img = (restored.numpy().squeeze() * 255).astype(np.uint8)
    restored_img = np.expand_dims(restored_img, axis=-1)
    save_img(output_path, restored_img)

input_folder = "Assets"
output_folder = "Restored_images"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        restore_image(os.path.join(input_folder, filename), os.path.join(output_folder, filename))
