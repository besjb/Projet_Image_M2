import os
from PIL import Image, ImageFilter, ImageEnhance

# Dossiers source et cible
source_folder = "Dataset"
target_folder = "Dataset_preprocessed"
os.makedirs(target_folder, exist_ok=True)

def resize_image(image, target_size=(256, 256), convert_gray=False):
    """Redimensionne et convertit l'image en niveaux de gris si spécifié."""
    if convert_gray:
        image = image.convert("L")
    return image.resize(target_size)

def simulate_degradation(image):
    """Simule la dégradation par ajout de bruit, flou, ou altération des couleurs."""
    # Applying a blur
    image = image.filter(ImageFilter.GaussianBlur(radius=1))
    # Reducing color quality by enhancing and then decreasing color
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(0.5)  # Adjust the factor to simulate fading
    return image

for filename in os.listdir(source_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(source_folder, filename)
        try:
            img = Image.open(img_path)
            # Decide whether to convert images to grayscale based on your model's requirements
            processed_image = resize_image(img, convert_gray=False)
            # Optionally simulate degradation if training requires such examples
            degraded_image = simulate_degradation(processed_image)
            
            processed_image.save(os.path.join(target_folder, f"processed_{filename}"))
            degraded_image.save(os.path.join(target_folder, f"degraded_{filename}"))
            print(f"Image prétraitée : {filename}")
        except Exception as e:
            print(f"Erreur pour {filename} : {e}")

print("Prétraitement terminé.")
