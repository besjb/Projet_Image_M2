import os
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw
import random

source_folder = 'Dataset/Real_Photos'
synthesized_folder = 'Dataset/Synthesized_Old_Photos'
old_photos_folder = 'Dataset/Old_Photos'

os.makedirs(synthesized_folder, exist_ok=True)
os.makedirs(old_photos_folder, exist_ok=True)

def add_fading(image):
    """Réduit le contraste et la saturation des couleurs pour simuler un effet de décoloration."""
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(0.5)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(0.5)
    return image

def add_stains(image):
    """Ajoute des taches aléatoires sur l'image."""
    draw = ImageDraw.Draw(image)
    for _ in range(random.randint(5, 10)):
        x0 = random.randint(0, image.width)
        y0 = random.randint(0, image.height)
        radius = random.randint(10, 30) 
        draw.ellipse((x0, y0, x0 + radius, y0 + radius), fill=None, outline=(128, 128, 128))
    return image

def add_scratches_and_tears(image):
    """Ajoute des rayures et des déchirures simulées sur l'image."""
    draw = ImageDraw.Draw(image)
    for _ in range(random.randint(5, 15)):
        x0 = random.randint(0, image.width)
        y0 = random.randint(0, image.height)
        x1 = x0 + random.randint(-50, 50)
        y1 = y0 + random.randint(-50, 50)
        draw.line((x0, y0, x1, y1), fill=(128, 128, 128), width=random.randint(1, 3))
    return image

def add_noise(image):
    """Ajoute du bruit aléatoire sur l'image."""
    np_image = np.array(image)
    noise = np.random.normal(0, 25, np_image.shape)
    noisy_image = np_image + noise
    noisy_image = np.clip(noisy_image, 0, 255)
    return Image.fromarray(noisy_image.astype('uint8'))

def process_and_save_images():
    """Applique les effets de dégradation, convertit en niveaux de gris et sauvegarde les images."""
    for filename in os.listdir(source_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(source_folder, filename)
            try:
                print(f"Processing file: {filename}")
                img = Image.open(img_path).convert('RGB')
                degraded_img = add_fading(img)
                degraded_img = add_stains(degraded_img)
                degraded_img = add_scratches_and_tears(degraded_img)
                degraded_img = add_noise(degraded_img)

                synthesized_path = os.path.join(synthesized_folder, f"degraded_{filename}")
                degraded_img.save(synthesized_path)
                print(f"Saved synthesized image: {synthesized_path}")

                grayscale_img = degraded_img.convert('L') 
                grayscale_path = os.path.join(old_photos_folder, f"gray_{filename}")
                grayscale_img.save(grayscale_path)
                print(f"Converted to grayscale and saved: {grayscale_path}")

                print(f"Image mode after grayscale conversion: {grayscale_img.mode}")
                if grayscale_img.mode != 'L':
                    raise ValueError(f"Failed to convert image to grayscale: {grayscale_path}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

process_and_save_images()
