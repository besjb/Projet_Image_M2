import os
import random
import numpy as np
from PIL import Image, ImageDraw

# Dossiers source et cible
source_folder = "Dataset"
target_folder = "Dataset_preprocessed"
asset_folder = "Assets"
asset_pre_folder = "Assets_preprocessed"

# Créer les sous-dossiers pour les images bruitées et dégradées
noisy_folder = os.path.join(target_folder, "noisy_images")
degraded_folder = os.path.join(target_folder, "degraded_images")

os.makedirs(noisy_folder, exist_ok=True)
os.makedirs(degraded_folder, exist_ok=True)

def resize_image(image, target_size=(256, 256)):
    """Redimensionne l'image à la taille cible."""
    return image.convert("L").resize(target_size)

def AjoutLignes(image, num_lines=5):
    """Ajoute des lignes aléatoires à l'image."""
    draw = ImageDraw.Draw(image)
    width, height = image.size

    for _ in range(num_lines):
        line_color = 255
        line_thickness = random.randint(1, 5)
        orientation = random.choice(["horizontal", "vertical", "diagonal"])

        if orientation == "horizontal":
            y = random.randint(0, height - 1)
            draw.line([(0, y), (width, y)], fill=line_color, width=line_thickness)
        elif orientation == "vertical":
            x = random.randint(0, width - 1)
            draw.line([(x, 0), (x, height)], fill=line_color, width=line_thickness)
        elif orientation == "diagonal":
            start_x = random.randint(0, width)
            start_y = random.randint(0, height)
            end_x = random.randint(0, width)
            end_y = random.randint(0, height)
            draw.line([(start_x, start_y), (end_x, end_y)], fill=line_color, width=line_thickness)
    return image

def preprocess_image_for_noise(image, target_size=(256, 256)):
    """Ajoute du bruit à une image."""
    image = resize_image(image, target_size)
    image_np = np.array(image)
    sigma = 20  # Intensité du bruit
    noise = np.random.normal(0, sigma, image_np.shape)
    noisy_image_np = image_np + noise
    noisy_image_np = np.clip(noisy_image_np, 0, 255)
    noisy_image = Image.fromarray(noisy_image_np.astype(np.uint8))
    return noisy_image

def preprocess_image_for_degradation(image, target_size=(256, 256)):
    """Ajoute des lignes aléatoires à une image."""
    image = resize_image(image, target_size)
    nbLignes = random.randint(1, 7)
    return AjoutLignes(image, nbLignes)

# Traiter toutes les images du dossier source
for filename in os.listdir(source_folder):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(source_folder, filename)
        img = Image.open(img_path)

        noisy_img = preprocess_image_for_noise(img)
        noisy_img.save(os.path.join(noisy_folder, filename))

        degraded_img = preprocess_image_for_degradation(img)
        degraded_img.save(os.path.join(degraded_folder, filename))

print("Prétraitement terminé !")