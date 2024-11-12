import os
import random
import math
from PIL import Image, ImageFilter, ImageDraw
import numpy as np

source_folder = "Dataset"
target_folder = "Dataset_preprocessed"

os.makedirs(target_folder, exist_ok=True)


def AjoutLignes(image, num_lines=5):
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
            if random.choice([True, False]):
                start_x = random.randint(0, width // 2)
                start_y = random.randint(0, height // 2)
                end_x = start_x + width
                end_y = start_y + height
            else:
                start_x = random.randint(0, width // 2)
                start_y = random.randint(height // 2, height)
                end_x = start_x + width
                end_y = start_y - height
            
            draw.line([(start_x, start_y), (end_x, end_y)], fill=line_color, width=line_thickness)
    return image


def preprocess_image(image):
    image = image.convert("L")
    image = image.resize((256, 256))
    image_np = np.array(image)

    sigma = 20
    noise = np.random.normal(0, sigma, image_np.shape)
    noisy_image_np = image_np + noise
    noisy_image_np = np.clip(noisy_image_np, 0, 255)
    noisy_image = Image.fromarray(noisy_image_np.astype(np.uint8))

    nbLignes  = random.randint(1, 7)
    degraded_image = AjoutLignes(noisy_image, nbLignes)

    return degraded_image

for filename in os.listdir(source_folder):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(source_folder, filename)
        img = Image.open(img_path)
        processed_img = preprocess_image(img)
        save_path = os.path.join(target_folder, filename)
        processed_img.save(save_path)

print("Prétraitements finis !")
