import os
from PIL import Image

# Dossiers source et cible
source_folder = "Dataset"
target_folder = "Dataset_preprocessed"

os.makedirs(target_folder, exist_ok=True)

def resize_image(image, target_size=(256, 256)):
    """Convertit l'image en niveaux de gris et redimensionne."""
    return image.convert("L").resize(target_size)

for filename in os.listdir(source_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(source_folder, filename)
        try:
            img = Image.open(img_path)
            gray_image = resize_image(img)
            gray_image.save(os.path.join(target_folder, filename))
            print(f"Image prétraitée : {filename}")
        except Exception as e:
            print(f"Erreur pour {filename} : {e}")

print("Prétraitement terminé.")
