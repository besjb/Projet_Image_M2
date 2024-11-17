import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
from vae_model import VAE

# Charger le modèle VAE
vae = VAE(latent_dim=256)
vae.load_weights("vae_weights_256x256_segmentation.weights.h5")

def generate_mask(image_path, output_path):
    img = load_img(image_path, target_size=(256, 256), color_mode="grayscale")
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    mask, _, _ = vae(img_array)
    
    mask_img = (mask.numpy().squeeze() * 255).astype(np.uint8)
    mask_img = np.expand_dims(mask_img, axis=-1)
    
    save_img(output_path, mask_img)

input_folder = "Assets"
output_folder = "Assets_masks"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        input_image_path = os.path.join(input_folder, filename)
        output_mask_path = os.path.join(output_folder, filename)
        
        generate_mask(input_image_path, output_mask_path)
