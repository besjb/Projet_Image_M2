import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
from tensorflow.keras.models import load_model
from model import MultiDomainVAE
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import argparse

IMAGE_SIZE = (256, 256)

INVERT = False
PONDERATION_X = 0.      # Domaines des images propres (ça doit donner des images lisses)
PONDERATION_Y = 0.2     # Domaine des images synthétiques (débruitage)
PONDERATION_Z = 0.8     # Domaine des images anciennes (réparation des dégradations)

def preprocess_image(image_path):
    """Charge et prétraite une image pour la passer dans le modèle."""
    try:
        image = load_img(image_path, target_size=IMAGE_SIZE, color_mode='grayscale')
        image_array = img_to_array(image) / 255.0   # Normalisation entre 0 et 1
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

def process_image(model, input_path, output_path):
    """Applique le modèle sur une seule image et sauvegarde le résultat."""
    print(f"Traitement de l'image : {input_path}")
    preprocessed_image = preprocess_image(input_path)

    if preprocessed_image is None:
        print(f"Impossible de prétraiter l'image : {input_path}")
        return

    inputs = {
        'X': preprocessed_image,                # Entrée réelle pour X
        'Y': create_placeholder(batch_size=1),  # Placeholder pour Y
        'Z': create_placeholder(batch_size=1)   # Placeholder pour Z
    }

    # Reconstruit les images avec le modèle
    outputs = model(inputs, training=False)

    reconstructed_X = outputs['X'].numpy()
    reconstructed_Y = outputs['Y'].numpy()
    reconstructed_Z = outputs['Z'].numpy()

    reconstructed_image = PONDERATION_Y * reconstructed_Y + PONDERATION_Z * reconstructed_Z

    if INVERT:
        reconstructed_image = 1. - reconstructed_image

    save_image(reconstructed_image[0], output_path)

    # Calcul des métriques PSNR et SSIM
    original_image = np.squeeze(preprocessed_image)         # Image originale
    restored_image = np.squeeze(reconstructed_image[0])     # Image restaurée

    psnr_value = psnr(original_image, restored_image, data_range=1.0)
    ssim_value = ssim(original_image, restored_image, data_range=1.0)

    print(f"PSNR: {psnr_value:.2f}, SSIM: {ssim_value:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Restauration d'une image ancienne avec un modèle IA")
    parser.add_argument("--model", type=str, required=True, help="Chemin du fichier de modèle")
    parser.add_argument("--input", type=str, required=True, help="Chemin de l'image à restaurer")
    parser.add_argument("--output", type=str, required=True, help="Chemin de l'image restaurée")
    args = parser.parse_args()

    model = load_model(args.model, custom_objects={"MultiDomainVAE": MultiDomainVAE})
    process_image(model, args.input, args.output)
    print("Traitement terminé.")
