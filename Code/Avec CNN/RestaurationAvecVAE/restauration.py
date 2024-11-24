import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
from tensorflow.keras.models import load_model
from model import MultiDomainVAE
import numpy as np

# Configurations
INPUT_FOLDER = "Assets"
OUTPUT_FOLDER = "Restaurés"
IMAGE_SIZE = (256, 256)
LATENT_DIM = 256
MODEL_WEIGHTS_PATH = "vae_final.weights.h5"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def preprocess_image(image_path):
    """Charge et prétraite une image pour la passer dans le modèle."""
    try:
        image = load_img(image_path, target_size=IMAGE_SIZE, color_mode='grayscale')
        image_array = img_to_array(image) / 255.0  # Normalisation entre 0 et 1
        return np.expand_dims(image_array, axis=0)  # Ajout d'une dimension batch
    except Exception as e:
        print(f"Erreur lors du prétraitement de {image_path} : {e}")
        return None

def save_image(image_array, output_path):
    """Sauvegarde une image normalisée reconstruite par le modèle."""
    try:
        image_array = np.clip(image_array * 255, 0, 255).astype("uint8")
        if len(image_array.shape) == 2:
            image_array = np.expand_dims(image_array, axis=-1)
        save_img(output_path, image_array)
        print(f"Image sauvegardée : {output_path}")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde de l'image {output_path} : {e}")

def create_placeholder(batch_size=1):
    """Crée un tenseur placeholder vide pour les domaines manquants."""
    return np.zeros((batch_size, IMAGE_SIZE[0], IMAGE_SIZE[1], 1), dtype=np.float32)

def process_images(model, input_folder, output_folder):
    """Applique le modèle sur les images d'un dossier et sauvegarde les résultats."""
    for file_name in os.listdir(input_folder):
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)

        if not file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            print(f"Fichier ignoré : {input_path}")
            continue

        print(f"Traitement de l'image : {input_path}")
        preprocessed_image = preprocess_image(input_path)

        if preprocessed_image is None:
            print(f"Impossible de prétraiter l'image : {input_path}")
            continue

        inputs = {
            'X': create_placeholder(batch_size=1),  # Entrée réelle pour X
            'Y': preprocessed_image,                # Placeholder pour Y
            'Z': create_placeholder(batch_size=1)   # Placeholder pour Z
        }

        # Reconstruit les images avec le modèle
        outputs = model(inputs, training=False)

        reconstructed_X = outputs['Z'].numpy()
        reconstructed_Y = outputs['Y'].numpy()

        # Combinaison des deux
        reconstructed_image = 0.2 * reconstructed_Y + 0.8 * reconstructed_X
        reconstructed_image = 1. - reconstructed_image
        save_image(reconstructed_image[0], output_path)

if __name__ == "__main__":
    try:
        model = load_model("vae_full_model.keras", custom_objects={"MultiDomainVAE": MultiDomainVAE})
        print(f"Poids du modèle chargés depuis '{MODEL_WEIGHTS_PATH}'.")
    except Exception as e:
        print(f"Erreur lors du chargement des poids : {e}")
        exit(1)

    process_images(model, INPUT_FOLDER, OUTPUT_FOLDER)
    print(f"Traitement terminé. Images restaurées dans {OUTPUT_FOLDER}.")
