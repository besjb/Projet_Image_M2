import os
import random
import numpy as np
from PIL import Image, ImageDraw

# Dossiers source et cible
source_folder = "Dataset"
target_folder = "Dataset_preprocessed"

# Créer les sous-dossiers pour les images dégradées et les masques
degraded_folder = os.path.join(target_folder, "degraded_images")
mask_folder = os.path.join(target_folder, "masks")

os.makedirs(degraded_folder, exist_ok=True)
os.makedirs(mask_folder, exist_ok=True)

def resize_image(image, target_size=(256, 256)):
    """Redimensionne l'image à la taille cible."""
    return image.convert("L").resize(target_size)

#Technique 3 (ligne "random" mais on fait une rotation a la fin du segment)

# import random
# from PIL import Image, ImageDraw

# def AjoutLignesAvecMasque(image, num_lines=5, max_deviation=5):
#     """Ajoute des lignes aléatoires bruitées continues à l'image et génère un masque correspondant."""
#     draw = ImageDraw.Draw(image)
#     width, height = image.size
#     mask = Image.new("L", (width, height), 0)  # Masque binaire initialisé à 0
#     mask_draw = ImageDraw.Draw(mask)

#     for _ in range(num_lines):
#         line_color = 255
#         line_thickness = random.randint(1, 15)
#         orientation = random.choice(["horizontal", "vertical", "diagonal"])
#         segments = random.randint(10, 30)  # Diviser la ligne en plusieurs segments

#         # Définir le point de départ initial
#         start_x = random.randint(0, width - 1)
#         start_y = random.randint(0, height - 1)

#         current_x, current_y = start_x, start_y  # Les coordonnées actuelles

#         for _ in range(segments):
#             if orientation == "horizontal":
#                 # Déviation horizontale uniquement (x change, y constant avec bruit)
#                 new_x = current_x + random.randint(5, 20)  # Longueur de segment
#                 new_y = current_y + random.randint(-max_deviation, max_deviation)  # Bruit vertical
#                 if new_x >= width: 
#                     new_x = width - 1
#                 draw.line([(current_x, current_y), (new_x, new_y)], fill=line_color, width=line_thickness)
#                 mask_draw.line([(current_x, current_y), (new_x, new_y)], fill=255, width=line_thickness)
#                 current_x, current_y = new_x, new_y
#                 if current_x >= width - 1:
#                     break 

#             elif orientation == "vertical":
#                 # Déviation verticale uniquement (y change, x constant avec bruit)
#                 new_y = current_y + random.randint(5, 20)  # Longueur de segment
#                 new_x = current_x + random.randint(-max_deviation, max_deviation)  # Bruit horizontal
#                 if new_y >= height: 
#                     new_y = height - 1
#                 draw.line([(current_x, current_y), (new_x, new_y)], fill=line_color, width=line_thickness)
#                 mask_draw.line([(current_x, current_y), (new_x, new_y)], fill=255, width=line_thickness)
#                 current_x, current_y = new_x, new_y
#                 if current_y >= height - 1:
#                     break 

#             elif orientation == "diagonal":
#                 # Déviation dans les deux directions
#                 new_x = current_x + random.randint(-20, 20)
#                 new_y = current_y + random.randint(-20, 20)
#                 if new_x < 0:
#                     new_x = 0
#                 if new_x >= width:
#                     new_x = width - 1
#                 if new_y < 0:
#                     new_y = 0
#                 if new_y >= height:
#                     new_y = height - 1
#                 draw.line([(current_x, current_y), (new_x, new_y)], fill=line_color, width=line_thickness)
#                 mask_draw.line([(current_x, current_y), (new_x, new_y)], fill=255, width=line_thickness)
#                 current_x, current_y = new_x, new_y
#                 if current_x >= width - 1 or current_y >= height - 1:
#                     break 

#     return image, mask


# Technique 2 (ligne "random" mais on fait une translation a la fin du segment)

# def AjoutLignesAvecMasque(image, num_lines=5, max_deviation=5):
#     """Ajoute des lignes aléatoires bruitées à l'image et génère un masque correspondant."""
#     draw = ImageDraw.Draw(image)
#     width, height = image.size
#     mask = Image.new("L", (width, height), 0)  # Masque binaire initialisé à 0
#     mask_draw = ImageDraw.Draw(mask)

#     for _ in range(num_lines):
#         line_color = 255
#         line_thickness = random.randint(1, 15)
#         orientation = random.choice(["horizontal", "vertical", "diagonal"])
#         segments = random.randint(10, 30)  # Diviser la ligne en plusieurs segments

#         if orientation == "horizontal":
#             y = random.randint(0, height - 1)
#             segment_length = width // segments
#             for i in range(segments):
#                 start_x = i * segment_length
#                 end_x = (i + 1) * segment_length
#                 noisy_y = y + random.randint(-max_deviation, max_deviation)
#                 draw.line([(start_x, noisy_y), (end_x, noisy_y)], fill=line_color, width=line_thickness)
#                 mask_draw.line([(start_x, noisy_y), (end_x, noisy_y)], fill=255, width=line_thickness)

#         elif orientation == "vertical":
#             x = random.randint(0, width - 1)
#             segment_length = height // segments
#             for i in range(segments):
#                 start_y = i * segment_length
#                 end_y = (i + 1) * segment_length
#                 noisy_x = x + random.randint(-max_deviation, max_deviation)
#                 draw.line([(noisy_x, start_y), (noisy_x, end_y)], fill=line_color, width=line_thickness)
#                 mask_draw.line([(noisy_x, start_y), (noisy_x, end_y)], fill=255, width=line_thickness)

#         elif orientation == "diagonal":
#             start_x = random.randint(0, width)
#             start_y = random.randint(0, height)
#             end_x = random.randint(0, width)
#             end_y = random.randint(0, height)
#             dx = (end_x - start_x) / segments
#             dy = (end_y - start_y) / segments
#             for i in range(segments):
#                 noisy_start_x = start_x + i * dx + random.uniform(-max_deviation, max_deviation)
#                 noisy_start_y = start_y + i * dy + random.uniform(-max_deviation, max_deviation)
#                 noisy_end_x = start_x + (i + 1) * dx + random.uniform(-max_deviation, max_deviation)
#                 noisy_end_y = start_y + (i + 1) * dy + random.uniform(-max_deviation, max_deviation)
#                 draw.line([(noisy_start_x, noisy_start_y), (noisy_end_x, noisy_end_y)], fill=line_color, width=line_thickness)
#                 mask_draw.line([(noisy_start_x, noisy_start_y), (noisy_end_x, noisy_end_y)], fill=255, width=line_thickness)

#     return image, mask

# technique 1 (ligne lineaire)

def AjoutLignesAvecMasque(image, num_lines=5):
    """Ajoute des lignes aléatoires à l'image et génère un masque correspondant."""
    draw = ImageDraw.Draw(image)
    width, height = image.size
    mask = Image.new("L", (width, height), 0)  # Masque binaire initialisé à 0
    mask_draw = ImageDraw.Draw(mask)

    for _ in range(num_lines):
        line_color = 255
        line_thickness = random.randint(1, 15)
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
