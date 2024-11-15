import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img

# Charger le modèle
model = load_model("models/unet_colorization.h5")

def preprocess_image(image_path, target_size=(256, 256)):
    img_gray = load_img(image_path, target_size=target_size, color_mode="grayscale")
    return np.expand_dims(img_to_array(img_gray) / 255.0, axis=0)  # Ajouter une dimension batch

def colorize_image(model, input_path, output_path):
    gray_image = preprocess_image(input_path)
    colorized_image = model.predict(gray_image)[0]
    colorized_image = (colorized_image * 255).astype(np.uint8)
    save_img(output_path, colorized_image)
    print(f"Image colorisée enregistrée dans : {output_path}")

# Coloriser toutes les images du dossier Assets
input_folder = "Assets"
output_folder = "Assets_preprocessed"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        colorize_image(model, input_path, output_path)
