import os
from PIL import Image, ImageDraw
import numpy as np
import random

input_folder = 'Dataset/Real_Photos'
output_folder_with_degradations = 'Dataset/Synthesized_Old_Photos'
output_folder_resized = 'Dataset/Old_Photos'

resize_target_size = (256, 256)
os.makedirs(output_folder_with_degradations, exist_ok=True)

def validate_and_clean_images(folder):
    """Valide les fichiers d'image et supprime ceux qui sont invalides."""
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    invalid_files = []
    
    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)
        if not any(file_name.lower().endswith(ext) for ext in valid_extensions):
            print(f"Fichier invalide trouvé : {file_path}")
            invalid_files.append(file_path)
            continue
        try:
            with Image.open(file_path) as image:
                image.verify()
        except Exception as e:
            print(f"Image corrompue trouvée : {file_path}, Erreur : {e}")
            invalid_files.append(file_path)
    
    for file_path in invalid_files:
        try:
            os.remove(file_path)
            print(f"Fichier supprimé : {file_path}")
        except Exception as e:
            print(f"Impossible de supprimer le fichier : {file_path}, Erreur : {e}")

def resize_image(image, size=(256, 256)):
    """Redimensionne l'image à la taille spécifiée."""
    try:
        if not callable(getattr(image, "resize", None)):
            raise TypeError(f"La méthode 'resize' n'est pas callable sur l'objet : {type(image)}")
        
        resized_image = image.resize(size, Image.ANTIALIAS)
        return resized_image
    except Exception as e:
        print(f"Erreur lors du redimensionnement : {e}")
        raise

def add_cracks_to_image(image):
    """Ajoute des craquelures simulées (gris clair à blanc) à l'image."""
    draw_on_image = ImageDraw.Draw(image)
    for _ in range(random.randint(5, 15)):  # Nombre aléatoire de craquelures
        start_x = random.randint(0, image.width)
        start_y = random.randint(0, image.height)
        end_x = start_x + random.randint(-50, 50)
        end_y = start_y + random.randint(-50, 50)
        draw_on_image.line((start_x, start_y, end_x, end_y), fill=random.randint(200, 255), width=random.randint(1, 3))
    return image

def add_stains_to_image(image):
    """Ajoute des taches simulées (gris clair à blanc) sur l'image."""
    draw_on_image = ImageDraw.Draw(image)
    for _ in range(random.randint(3, 10)):  # Nombre aléatoire de taches
        center_x = random.randint(0, image.width)
        center_y = random.randint(0, image.height)
        radius = random.randint(10, 30)
        draw_on_image.ellipse((center_x, center_y, center_x + radius, center_y + radius), fill=random.randint(200, 255))
    return image

def add_noise_to_image(image):
    """Ajoute du bruit aléatoire sur l'image."""
    np_image = np.array(image)
    noise = np.random.normal(0, 3, np_image.shape)  # Bruit gaussien
    noisy_image = np.clip(np_image + noise, 0, 255).astype('uint8')
    return Image.fromarray(noisy_image)

def apply_degradations_to_image(image):
    """Applique une série de dégradations naturelles à l'image."""
    image = add_cracks_to_image(image)
    image = add_stains_to_image(image)
    image = add_noise_to_image(image)
    return image

def process_images(input_folder, output_folder=None, size=(256, 256), apply_degradations=False):
    """
    Parcourt les images du dossier source, applique les dégradations si demandé, redimensionne, et sauvegarde.
    Si `output_folder` est None, les images seront sauvegardées dans leur dossier d'origine.
    """
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        try:
            with Image.open(file_path) as image:
                print(f"Traitement de l'image : {file_path}")
                # Conversion en NDG
                if image.mode != 'L':
                    image = image.convert('L')
                # Redimensionnement
                image = resize_image(image, size=size)
                if apply_degradations:
                    image = apply_degradations_to_image(image)

                output_path = os.path.join(output_folder or input_folder, file_name)
                image.save(output_path)
                print(f"Image traitée et sauvegardée : {output_path}")
        except Exception as e:
            print(f"Erreur avec le fichier {file_path} : {e}")

# Nettoyage des dossiers
validate_and_clean_images(input_folder)
validate_and_clean_images(output_folder_resized)

# Redimensionnement des images et convertion en NDG
process_images(output_folder_resized, size=resize_target_size, apply_degradations=False)

# Application des dégradations aux images de `Real_Photos` et les sauvegarder dans `Synthesized_Old_Photos`
process_images(input_folder, output_folder_with_degradations, size=resize_target_size, apply_degradations=True)
