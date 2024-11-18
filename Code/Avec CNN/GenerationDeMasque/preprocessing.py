import os
import random
import numpy as np
from PIL import Image, ImageDraw

# Dossiers source et cible
source_folder = "../ColorisationUNET/Dataset"
target_folder = "Dataset_preprocessed"

# Créer les sous-dossiers pour les images dégradées et les masques
degraded_folder = os.path.join(target_folder, "degraded_images")
mask_folder = os.path.join(target_folder, "masks")

os.makedirs(degraded_folder, exist_ok=True)
os.makedirs(mask_folder, exist_ok=True)

def resize_image(image, target_size=(256, 256)):
    """Redimensionne l'image à la taille cible."""
    return image.convert("L").resize(target_size)

def AjoutLignesAvecMasque(image, num_lines=5):
    """Ajoute des lignes aléatoires à l'image et génère un masque correspondant."""
    draw = ImageDraw.Draw(image)
    width, height = image.size
    mask = Image.new("L", (width, height), 0)  # Masque binaire initialisé à 0
    mask_draw = ImageDraw.Draw(mask)

    for _ in range(num_lines):
        line_color = 255
        line_thickness = random.randint(1, 5)
        orientation = random.choice(["horizontal", "vertical", "diagonal"])

        if orientation == "horizontal":
            y = random.randint(0, height - 1)
            draw.line([(0, y), (width, y)], fill=line_color, width=line_thickness)
            mask_draw.line([(0, y), (width, y)], fill=255, width=line_thickness)
        elif orientation == "vertical":
            x = random.randint(0, width - 1)
            draw.line([(x, 0), (x, height)], fill=line_color, width=line_thickness)
            mask_draw.line([(x, 0), (x, height)], fill=255, width=line_thickness)
        elif orientation == "diagonal":
            start_x = random.randint(0, width)
            start_y = random.randint(0, height)
            end_x = random.randint(0, width)
            end_y = random.randint(0, height)
            draw.line([(start_x, start_y), (end_x, end_y)], fill=line_color, width=line_thickness)
            mask_draw.line([(start_x, start_y), (end_x, end_y)], fill=255, width=line_thickness)
    return image, mask

def preprocess_image_for_degradation(image, target_size=(256, 256)):
    """Ajoute des lignes aléatoires à une image et génère un masque."""
    image = resize_image(image, target_size)
    nbLignes = random.randint(1, 7)
    return AjoutLignesAvecMasque(image, nbLignes)

# Traiter toutes les images du dossier source
for filename in os.listdir(source_folder):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(source_folder, filename)
        img = Image.open(img_path)

        # Prétraitement dégradé avec masque
        degraded_img, mask_img = preprocess_image_for_degradation(img)
        degraded_img.save(os.path.join(degraded_folder, filename))
        mask_img.save(os.path.join(mask_folder, filename))

print("Prétraitement terminé avec masques générés !")
