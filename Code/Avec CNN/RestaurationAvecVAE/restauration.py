import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
from PIL import Image
from model import MultiDomainVAE 

def validate_and_clean_images(directory):
    """Valide les fichiers d'image et supprime ceux qui sont invalides."""
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if not any(file.lower().endswith(ext) for ext in valid_extensions):
                print(f"Invalid file found and removed: {file_path}")
                os.remove(file_path)
                continue
            try:
                with Image.open(file_path) as img:
                    img.verify() 
            except Exception as e:
                print(f"Corrupted image found and removed: {file_path}, Error: {e}")
                os.remove(file_path)

def load_and_preprocess_image(image_path, target_size=(256, 256)):
    """Charge et prétraite une image pour la restauration."""
    img = load_img(image_path, target_size=target_size, color_mode='rgb')
    img_array = img_to_array(img)
    img_array /= 255.0
    return np.expand_dims(img_array, axis=0)

def save_image(image_array, output_path):
    """Sauvegarde un tableau numpy en tant qu'image."""
    if image_array.ndim == 3 and image_array.shape[-1] == 3:
        image_array = np.clip(image_array * 255, 0, 255).astype('uint8')
    else:
        print(f"Unexpected image shape {image_array.shape}, skipping save.")
        return
    save_img(output_path, image_array)

def restore_images(input_dir, output_dir, model):
    """Restaure les images depuis un dossier."""
    validate_and_clean_images(input_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(input_dir, filename)
            processed_image = load_and_preprocess_image(file_path)
            
            if processed_image.shape[1:] != (256, 256, 3):
                print(f"Skipping {filename}: unexpected shape {processed_image.shape}")
                continue

            restored_image = model(processed_image, training=False)
            restored_image = restored_image.numpy().squeeze()
            
            save_image(restored_image, os.path.join(output_dir, filename))
            print(f"Restored image saved: {filename}")

if __name__ == "__main__":
    model = MultiDomainVAE(latent_dim=256) 
    model.build(input_shape=(None, 256, 256, 3))
    model.load_weights('best_model_weights.weights.h5')
    print("Poids chargés avec succès.")

    input_images_dir = 'Assets'
    output_images_dir = 'Restored_images'

    restore_images(input_images_dir, output_images_dir, model)
    print(f"Restauration terminée. Images sauvegardées dans {output_images_dir}.")
