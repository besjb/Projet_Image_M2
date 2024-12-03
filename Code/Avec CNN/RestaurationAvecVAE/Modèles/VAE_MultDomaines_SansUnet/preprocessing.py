import os
import numpy as np
import random
from PIL import Image, ImageDraw

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

        resized_image = image.resize(size, Image.Resampling.LANCZOS)
        return resized_image
    except Exception as e:
        print(f"Erreur lors du redimensionnement : {e}")
        raise
        
def AjoutLignes(image, num_lines=5, max_deviation=5):
    """
    Ajoute des lignes aléatoires bruitées continues à l'image.
    """
    # Validation du type et du mode de l'image
    if not isinstance(image, Image.Image):
        raise ValueError("L'entrée doit être une instance de PIL.Image.Image")
    
    if image.mode not in ["L", "RGB"]:
        print(f"Conversion du mode {image.mode} au mode 'L'.")
        image = image.convert("L")

    width, height = image.size
    draw = ImageDraw.Draw(image)

    for _ in range(num_lines):
        line_color = 255  # Blanc
        line_thickness = random.randint(1, 15)
        orientation = random.choice(["horizontal", "vertical", "diagonal"])
        segments = random.randint(10, 30)  # Nombre de segments

        # Point de départ initial
        start_x = random.randint(0, width - 1)
        start_y = random.randint(0, height - 1)
        current_x, current_y = start_x, start_y

        for _ in range(segments):
            if orientation == "horizontal":
                new_x = min(width - 1, current_x + random.randint(5, 20))
                new_y = max(0, min(height - 1, current_y + random.randint(-max_deviation, max_deviation)))
            elif orientation == "vertical":
                new_y = min(height - 1, current_y + random.randint(5, 20))
                new_x = max(0, min(width - 1, current_x + random.randint(-max_deviation, max_deviation)))
            elif orientation == "diagonal":
                new_x = max(0, min(width - 1, current_x + random.randint(-20, 20)))
                new_y = max(0, min(height - 1, current_y + random.randint(-20, 20)))

            # Dessiner les lignes sur l'image
            draw.line([(current_x, current_y), (new_x, new_y)], fill=line_color, width=line_thickness)

            current_x, current_y = new_x, new_y
            if current_x >= width - 1 or current_y >= height - 1:
                break

    return image
    
def add_noise_to_image(image):
    """Ajoute du bruit aléatoire sur l'image."""
    np_image = np.array(image)
    noise = np.random.normal(0, 3, np_image.shape)  # Bruit gaussien
    noisy_image = np.clip(np_image + noise, 0, 255).astype('uint8')
    return Image.fromarray(noisy_image)

def apply_degradations_to_image(image):
    """Applique une série de dégradations naturelles à l'image."""
    image = AjoutLignes(image)
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
