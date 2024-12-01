import os
import sys
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import subprocess
import importlib.util

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
from tensorflow.keras.models import load_model
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tkinter import Tk, filedialog
from PIL import Image, ImageTk


ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.geometry("1280x720")
app.title("Projet image - Restauration d'images anciennes")
app.configure(fg_color="#222222")

# Variables globales
original_image = None
processed_image = None
mask_image = None
mask = None
eroded_mask = None  # Masque érodé
apply_erosion = False
inpainting_radius = 3
brush_size = 15
selected_method = "Classique"
drawing = False 

# Widgets dynamiques
draw_button = None
load_mask_button = None
brush_size_slider = None
brush_size_label = None
brush_size_value = None
erosion_checkbox = None

selected_model = "Modèle 1"
IMAGE_SIZE = (256, 256)
PONDERATION_Y = 0.5  # Pondération pour le domaine synthétique (débruitage)
PONDERATION_Z = 0.5  # Pondération pour le domaine ancien (restauration)
INVERT = True

pond_X_slider = None
pond_Y_slider = None
pond_Z_slider = None
use_X_checkbox = None
use_Y_checkbox = None
use_Z_checkbox = None

def update_pond_X(value):
    global pond_X, pond_X_value
    pond_X = float(value)
    if pond_X_value: 
        pond_X_value.configure(text=f"{pond_X:.2f}")

def update_pond_Y(value):
    global pond_Y, pond_Y_value
    pond_Y = float(value)
    if pond_Y_value:
        pond_Y_value.configure(text=f"{pond_Y:.2f}")

def update_pond_Z(value):
    global pond_Z, pond_Z_value
    pond_Z = float(value)
    if pond_Z_value:
        pond_Z_value.configure(text=f"{pond_Z:.2f}")

def toggle_use_X():
    global use_X
    use_X = not use_X

def toggle_use_Y():
    global use_Y
    use_Y = not use_Y

def toggle_use_Z():
    global use_Z
    use_Z = not use_Z


# Variables globales pour les domaines
use_X = True  # Inclure Domaine X activé par défaut
use_Y = True  # Inclure Domaine Y activé par défaut
use_Z = True  # Inclure Domaine Z activé par défaut
pond_X = 0.5  # Pondération par défaut pour Domaine X
pond_Y = 0.2  # Pondération par défaut pour Domaine Y
pond_Z = 0.8  # Pondération par défaut pour Domaine Z

pond_X_value = None
pond_Y_value = None
pond_Z_value = None
pond_X_label = None
pond_Y_label = None
pond_Z_label = None

domains_label = None

model_directory = {
    "Modèle 1": "/home/evan/Téléchargements/Projet_Image_M2-main/Code/Avec CNN/Application (copie)/Modèles/VAE_MultDomaines_SansUnet",
    "Modèle 2": "/home/evan/Téléchargements/Projet_Image_M2-main/Code/Avec CNN/Application (copie)/Modèles/VAE_MultDomaines_AvecUnet",
    "Modèle 3": "/home/evan/Téléchargements/Projet_Image_M2-main/Code/Avec CNN/Application (copie)/Modèles/VAE_MultDomaines_Microsoft"
}

selected_model_file = None
model_file_selector = None

def import_model_module(model_path):
    """Importe dynamiquement le fichier model.py depuis le chemin donné."""
    try:
        spec = importlib.util.spec_from_file_location("model", os.path.join(model_path, "model.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"Erreur lors de l'importation de model.py : {e}")
        return None

def load_image():
    global original_image, mask, eroded_mask
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg *.bmp")])
    if file_path:
        original_image = Image.open(file_path).convert("L")
        mask = np.ones((original_image.height, original_image.width), dtype=np.uint8) * 255
        eroded_mask = None  # Réinitialiser le masque érodé
        display_image(original_image, original_label)

def load_mask():
    global mask, mask_image, eroded_mask
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")])
    if file_path:
        mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.bitwise_not(mask)  # Inverse les couleurs du masque
        eroded_mask = None  # Réinitialiser le masque érodé
        mask_image = Image.fromarray(mask)
        display_image(mask_image, mask_label)

def display_image(image, label_widget):
    if image:
        img_resized = image.resize((256, 256))
        img_tk = ImageTk.PhotoImage(img_resized)
        label_widget.configure(image=img_tk, text="")
        label_widget.image = img_tk
    else:
        label_widget.configure(image=None, text="")

def update_brush_size(value):
    global brush_size
    brush_size = int(float(value))
    brush_size_value.configure(text=f"{brush_size}")

def update_inpainting_radius(value):
    global inpainting_radius
    inpainting_radius = int(float(value))
    inpainting_radius_value.configure(text=f"{inpainting_radius}")

def toggle_erosion(state=None):
    """
    Active ou désactive l'érosion.
    :param state: booléen indiquant l'état du checkbox (True ou False)
    """
    global apply_erosion, eroded_mask
    if state is not None:
        apply_erosion = state
    else:
        apply_erosion = erosion_checkbox.get()
    eroded_mask = None  # Réinitialiser le masque érodé

def draw_mask():
    global mask, original_image, drawing, mask_image
    if original_image is None:
        ctk.CTkMessagebox.show_error("Erreur", "Veuillez charger une image d'abord !")
        return

    damaged_img = np.array(original_image)
    damaged_img_color = cv2.cvtColor(damaged_img, cv2.COLOR_GRAY2BGR)

    def draw(event, x, y, flags, param):
        global mask, drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                cv2.circle(mask, (x, y), brush_size, (0, 0, 0), -1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

    cv2.namedWindow("Dessiner le masque")
    cv2.setMouseCallback("Dessiner le masque", draw)

    while True:
        mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined_display = cv2.addWeighted(damaged_img_color, 0.5, mask_display, 0.5, 0)
        cv2.imshow("Dessiner le masque", combined_display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            mask_image = Image.fromarray(mask)
            display_image(mask_image, mask_label)
            cv2.destroyWindow("Dessiner le masque")
            break
        elif key == ord('q'):
            cv2.destroyWindow("Dessiner le masque")
            return

def apply_restoration():
    global original_image, processed_image, mask, eroded_mask
    if original_image is None:
        messagebox.showerror("Erreur", "Veuillez charger une image d'abord !")
        return

    if selected_method == "Classique":
        if mask is None:
            messagebox.showerror("Erreur", "Veuillez dessiner un masque pour la méthode Classique !")
            return

        damaged_img = np.array(original_image)
        mask_inverted = cv2.bitwise_not(mask)

        inpainted_img = cv2.inpaint(damaged_img, mask_inverted, inpaintRadius=inpainting_radius, flags=cv2.INPAINT_TELEA)
        expanded_image = dynamic_range_expansion(inpainted_img)

        psnr_value = calculate_psnr(damaged_img, expanded_image)
        ssim_value = calculate_ssim(damaged_img, expanded_image)

        processed_image = Image.fromarray(expanded_image)
        display_image(processed_image, processed_label)
        metrics_label.configure(
            text=f"PSNR: {psnr_value:.2f} dB | SSIM: {ssim_value:.3f}"
        )

    elif selected_method == "Hybride":
        if mask is None:
            messagebox.showerror("Erreur", "Veuillez charger un masque pour la méthode Hybride !")
            return

        # Appliquer l'érosion
        kernel = np.ones((3, 3), np.uint8)
        eroded_mask = cv2.erode(mask, kernel, iterations=1) if apply_erosion else mask

        # Traitement pour la méthode Hybride
        damaged_img = np.array(original_image)
        mask_inverted = cv2.bitwise_not(eroded_mask)

        inpainted_img = cv2.inpaint(damaged_img, mask_inverted, inpaintRadius=inpainting_radius, flags=cv2.INPAINT_TELEA)
        expanded_image = dynamic_range_expansion(inpainted_img)

        psnr_value = calculate_psnr(damaged_img, expanded_image)
        ssim_value = calculate_ssim(damaged_img, expanded_image)

        processed_image = Image.fromarray(expanded_image)
        display_image(processed_image, processed_label)
        metrics_label.configure(
            text=f"PSNR: {psnr_value:.2f} dB | SSIM: {ssim_value:.3f}"
        )

    elif selected_method == "IA":
        if not selected_model_file:
            messagebox.showerror("Erreur", "Aucun fichier modèle sélectionné. Veuillez en choisir un.")
            return

        try:
            model_path = os.path.join(model_directory[selected_model])
            model_module = import_model_module(model_path)
            MultiDomainVAE = model_module.MultiDomainVAE
            model = tf.keras.models.load_model(os.path.join(model_path, selected_model_file), custom_objects={"MultiDomainVAE": MultiDomainVAE})

            preprocessed_image = preprocess_image(original_image)
            if preprocessed_image is None:
                messagebox.showerror("Erreur", "Erreur lors du prétraitement de l'image.")
                return

            inputs = {}
            inputs['X'] = preprocessed_image if use_X else np.zeros_like(preprocessed_image)
            inputs['Y'] = preprocessed_image if use_Y else np.zeros_like(preprocessed_image)
            inputs['Z'] = preprocessed_image if use_Z else np.zeros_like(preprocessed_image)

            outputs = model(inputs, training=False)
            reconstructed_X = outputs.get('X', np.zeros_like(preprocessed_image)).numpy()
            reconstructed_Y = outputs.get('Y', np.zeros_like(preprocessed_image)).numpy()
            reconstructed_Z = outputs.get('Z', np.zeros_like(preprocessed_image)).numpy()

            reconstructed_image = pond_X * reconstructed_X + pond_Y * reconstructed_Y + pond_Z * reconstructed_Z
            if INVERT:
                reconstructed_image = 1.0 - reconstructed_image

            save_image(reconstructed_image[0], "/tmp/restored_image.png")
            processed_image = Image.open("/tmp/restored_image.png")
            display_image(processed_image, processed_label)
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la restauration : {e}")


def preprocess_image(pil_image):
    """Convertit une image PIL en tenseur normalisé pour le modèle."""
    try:
        image = pil_image.resize(IMAGE_SIZE).convert("L")  # Redimensionner et passer en niveaux de gris
        image_array = np.array(image) / 255.0  # Normalisation entre 0 et 1
        return np.expand_dims(np.expand_dims(image_array, axis=0), axis=-1)  # (1, 256, 256, 1)
    except Exception as e:
        print(f"Erreur lors du prétraitement de l'image : {e}")
        return None

def save_image(image_array, output_path):
    """Sauvegarde une image normalisée reconstruite par le modèle."""
    try:
        image_array = np.clip(image_array * 255, 0, 255).astype("uint8")
        image = Image.fromarray(image_array.squeeze(), mode="L")
        image.save(output_path)
    except Exception as e:
        print(f"Erreur lors de la sauvegarde de l'image : {e}")

# Fonction d'expansion dynamique
def dynamic_range_expansion(image, y_min=0, y_max=255):
    x_min = np.min(image)
    x_max = np.max(image)
    alpha = (y_min * x_max - y_max * x_min) / (x_max - x_min)
    beta = (y_max - y_min) / (x_max - x_min)
    expanded_image = alpha + beta * image
    return np.clip(expanded_image, y_min, y_max).astype(np.uint8)

# Calcul de PSNR
def calculate_psnr(original, restored):
    # Assurez-vous que les deux images sont de type uint8
    if original.dtype != np.uint8:
        if original.max() <= 1.0:  # Normalisé entre 0 et 1
            original = (original * 255).astype("uint8")
        else:
            original = original.astype("uint8")
    if restored.dtype != np.uint8:
        if restored.max() <= 1.0:  # Normalisé entre 0 et 1
            restored = (restored * 255).astype("uint8")
        else:
            restored = restored.astype("uint8")
    
    if original.shape != restored.shape:
        restored = cv2.resize(restored, (original.shape[1], original.shape[0]))

    return cv2.PSNR(original, restored)

# Calcul de SSIM
def calculate_ssim(original, restored):
    K1, K2, L = 0.01, 0.03, 255
    mu_x, mu_y = np.mean(original), np.mean(restored)
    sigma_x_sq, sigma_y_sq = np.var(original), np.var(restored)
    sigma_xy = np.cov(original.flatten(), restored.flatten())[0, 1]
    C1, C2 = (K1 * L) ** 2, (K2 * L) ** 2
    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x**2 + mu_y**2 + C1) * (sigma_x_sq + sigma_y_sq + C2)
    return numerator / denominator

def reset_displayed_images():
    global mask, mask_image, processed_image
    mask = None
    mask_image = None
    processed_image = None
    display_image(None, mask_label)
    display_image(None, processed_label)

# Fonction pour lister les fichiers disponibles
def list_model_files(model_type):
    global selected_model_file, model_file_selector
    directory = model_directory.get(model_type, "")
    if os.path.exists(directory):
        return [f for f in os.listdir(directory) if f.endswith(".keras") or f.endswith(".weights.h5")]
    return []

# Fonction pour définir le fichier de modèle sélectionné
def set_model_file(value):
    """Met à jour le fichier modèle sélectionné."""
    global selected_model_file
    selected_model_file = value
    print(f"Fichier de modèle sélectionné : {selected_model_file}")

# Modification de set_model pour afficher les fichiers disponibles dans un second dropdown
def set_model(value):
    global selected_model, model_file_selector, selected_model_file
    selected_model = value
    print(f"Modèle sélectionné : {selected_model}")

    if model_file_selector:
        model_file_selector.destroy()
        model_file_selector = None

    files = list_model_files(selected_model)
    if files:
        selected_model_file = files[0] 
        model_file_selector = ctk.CTkOptionMenu(app, values=files, command=set_model_file, width=300)
        model_file_selector.set(selected_model_file)
        model_file_selector.place(relx=0.5, rely=0.4, anchor="center")
        print(f"Fichier de modèle par défaut : {selected_model_file}")
    else:
        selected_model_file = None
        print(f"Aucun fichier de modèle trouvé dans le répertoire : {model_directory.get(selected_model, '')}")


# Mise à jour des widgets selon la méthode sélectionnée
def set_method(value):
    global selected_method, draw_button, load_mask_button, erosion_checkbox
    global brush_size_label, brush_size_slider, brush_size_value
    global inpainting_radius_label, inpainting_radius_slider, inpainting_radius_value
    global model_selector, model_file_selector
    global pond_X_slider, pond_Y_slider, pond_Z_slider
    global pond_X_label, pond_Y_label, pond_Z_label
    global use_X_checkbox, use_Y_checkbox, use_Z_checkbox
    global domains_label
    global pond_X_value, pond_Y_value, pond_Z_value
    global mask, mask_image, original_image

    processed_image = None
    display_image(None, processed_label)

    if original_image is not None:
        mask = np.ones((original_image.height, original_image.width), dtype=np.uint8) * 255
        mask_image = Image.fromarray(mask)
        display_image(mask_image, mask_label)

    widgets = [
        draw_button, load_mask_button, erosion_checkbox, brush_size_label, brush_size_slider, brush_size_value,
        inpainting_radius_label, inpainting_radius_slider, inpainting_radius_value, model_selector,
        model_file_selector, pond_X_slider, pond_Y_slider, pond_Z_slider,
        pond_X_value, pond_Y_value, pond_Z_value,
        pond_X_label, pond_Y_label, pond_Z_label,
        use_X_checkbox, use_Y_checkbox, use_Z_checkbox,
        domains_label
    ]

    for widget in widgets:
        if widget:
            widget.destroy()

    # Réinitialiser les références
    draw_button = load_mask_button = erosion_checkbox = None
    brush_size_label = brush_size_slider = brush_size_value = None
    inpainting_radius_label = inpainting_radius_slider = inpainting_radius_value = None
    model_selector = model_file_selector = None
    pond_X_slider = pond_Y_slider = pond_Z_slider = None
    pond_X_value = pond_Y_value = pond_Z_value = None
    pond_X_label = pond_Y_label = pond_Z_label = None
    use_X_checkbox = use_Y_checkbox = use_Z_checkbox = None

    # Définir la méthode sélectionnée
    selected_method = value

    if value == "Classique":
        # Ajouter les éléments spécifiques à la méthode Classique
        draw_button = ctk.CTkButton(app, text="Dessiner le masque", command=draw_mask, width=150)
        draw_button.place(relx=0.5, rely=0.3, anchor="center")

        brush_size_label = ctk.CTkLabel(app, text="Taille du pinceau", text_color="white")
        brush_size_label.place(relx=0.37, rely=0.78, anchor="center")
        brush_size_slider = ctk.CTkSlider(app, from_=5, to=50, command=update_brush_size, width=200)
        brush_size_slider.set(brush_size)
        brush_size_slider.place(relx=0.5, rely=0.78, anchor="center")
        brush_size_value = ctk.CTkLabel(app, text=f"{brush_size}", text_color="white")
        brush_size_value.place(relx=0.6, rely=0.78, anchor="center")

        inpainting_radius_label = ctk.CTkLabel(app, text="Rayon d'inpainting", text_color="white")
        inpainting_radius_label.place(relx=0.37, rely=0.85, anchor="center")
        inpainting_radius_slider = ctk.CTkSlider(app, from_=1, to=50, command=update_inpainting_radius, width=200)
        inpainting_radius_slider.set(inpainting_radius)
        inpainting_radius_slider.place(relx=0.5, rely=0.85, anchor="center")
        inpainting_radius_value = ctk.CTkLabel(app, text=f"{inpainting_radius}", text_color="white")
        inpainting_radius_value.place(relx=0.6, rely=0.85, anchor="center")

    elif value == "Hybride":
        # Ajouter les éléments spécifiques à la méthode Hybride
        load_mask_button = ctk.CTkButton(app, text="Charger un masque", command=load_mask, width=150)
        load_mask_button.place(relx=0.5, rely=0.3, anchor="center")

        erosion_checkbox = ctk.CTkCheckBox(app, text="Appliquer une érosion", command=lambda: toggle_erosion(erosion_checkbox.get()))
        erosion_checkbox.place(relx=0.5, rely=0.78, anchor="center")

        inpainting_radius_label = ctk.CTkLabel(app, text="Rayon d'inpainting", text_color="white")
        inpainting_radius_label.place(relx=0.37, rely=0.85, anchor="center")
        inpainting_radius_slider = ctk.CTkSlider(app, from_=1, to=50, command=update_inpainting_radius, width=200)
        inpainting_radius_slider.set(inpainting_radius)
        inpainting_radius_slider.place(relx=0.5, rely=0.85, anchor="center")
        inpainting_radius_value = ctk.CTkLabel(app, text=f"{inpainting_radius}", text_color="white")
        inpainting_radius_value.place(relx=0.6, rely=0.85, anchor="center")

    elif value == "IA":
        # Ajouter le dropdown pour sélectionner le modèle
        model_selector = ctk.CTkOptionMenu(app, values=["Modèle 1", "Modèle 2", "Modèle 3"], command=set_model, width=200)
        model_selector.place(relx=0.5, rely=0.3, anchor="center")
        set_model("Modèle 1")

        # Ajouter texte pour les domaines d'entrées
        domains_label = ctk.CTkLabel(app, text="Domaines d'entrées (XYZ)", text_color="white", font=("Arial", 14))
        domains_label.place(relx=0.5, rely=0.45, anchor="center")

        # Ajouter checkboxes pour les domaines
        use_X_checkbox = ctk.CTkCheckBox(app, text="Domaine X", command=toggle_use_X)
        use_X_checkbox.place(relx=0.4, rely=0.5, anchor="center")
        use_X_checkbox.select()

        use_Y_checkbox = ctk.CTkCheckBox(app, text="Domaine Y", command=toggle_use_Y)
        use_Y_checkbox.place(relx=0.5, rely=0.5, anchor="center")
        use_Y_checkbox.select()

        use_Z_checkbox = ctk.CTkCheckBox(app, text="Domaine Z", command=toggle_use_Z)
        use_Z_checkbox.place(relx=0.6, rely=0.5, anchor="center")
        use_Z_checkbox.select()

        pond_X_label = ctk.CTkLabel(app, text="Pondération X", text_color="white")
        pond_X_label.place(relx=0.4, rely=0.55, anchor="center")  # Était 0.5
        pond_X_slider = ctk.CTkSlider(app, from_=0.0, to=1.0, command=update_pond_X, width=120)
        pond_X_slider.set(pond_X)
        pond_X_slider.place(relx=0.4, rely=0.6, anchor="center")  # Était 0.55
        pond_X_value = ctk.CTkLabel(app, text=f"{pond_X:.2f}", text_color="white")
        pond_X_value.place(relx=0.4, rely=0.63, anchor="center")  # Était 0.58

        pond_Y_label = ctk.CTkLabel(app, text="Pondération Y", text_color="white")
        pond_Y_label.place(relx=0.5, rely=0.55, anchor="center")  # Était 0.5
        pond_Y_slider = ctk.CTkSlider(app, from_=0.0, to=1.0, command=update_pond_Y, width=120)
        pond_Y_slider.set(pond_Y)
        pond_Y_slider.place(relx=0.5, rely=0.6, anchor="center")  # Était 0.55
        pond_Y_value = ctk.CTkLabel(app, text=f"{pond_Y:.2f}", text_color="white")
        pond_Y_value.place(relx=0.5, rely=0.63, anchor="center")  # Était 0.58

        pond_Z_label = ctk.CTkLabel(app, text="Pondération Z", text_color="white")
        pond_Z_label.place(relx=0.6, rely=0.55, anchor="center")  # Était 0.5
        pond_Z_slider = ctk.CTkSlider(app, from_=0.0, to=1.0, command=update_pond_Z, width=120)
        pond_Z_slider.set(pond_Z)
        pond_Z_slider.place(relx=0.6, rely=0.6, anchor="center")  # Était 0.55
        pond_Z_value = ctk.CTkLabel(app, text=f"{pond_Z:.2f}", text_color="white")
        pond_Z_value.place(relx=0.6, rely=0.63, anchor="center")  # Était 0.58

model_selector = None

method_selector = ctk.CTkOptionMenu(app, values=["Classique", "Hybride", "IA"], command=set_method, width=200)
method_selector.place(relx=0.5, rely=0.1, anchor="center")

load_button = ctk.CTkButton(app, text="Charger", command=load_image, width=150)
load_button.place(relx=0.25, rely=0.3, anchor="center")

process_button = ctk.CTkButton(app, text="Appliquer la restauration", command=apply_restoration, width=200)
process_button.place(relx=0.75, rely=0.3, anchor="center")

inpainting_radius_label = ctk.CTkLabel(app, text="Rayon d'inpainting", text_color="white")
inpainting_radius_label.place(relx=0.37, rely=0.85, anchor="center")

inpainting_radius_slider = ctk.CTkSlider(app, from_=1, to=50, command=update_inpainting_radius, width=200)
inpainting_radius_slider.set(inpainting_radius)
inpainting_radius_slider.place(relx=0.5, rely=0.85, anchor="center")

inpainting_radius_value = ctk.CTkLabel(app, text=f"{inpainting_radius}", text_color="white")
inpainting_radius_value.place(relx=0.6, rely=0.85, anchor="center")

original_label = ctk.CTkLabel(app, text="")
original_label.place(relx=0.25, rely=0.55, anchor="center")

mask_label = ctk.CTkLabel(app, text="")
mask_label.place(relx=0.5, rely=0.55, anchor="center")

processed_label = ctk.CTkLabel(app, text="")
processed_label.place(relx=0.75, rely=0.55, anchor="center")

metrics_label = ctk.CTkLabel(app, text="", text_color="white", font=("Poppins", 16))
metrics_label.place(relx=0.5, rely=0.95, anchor="center")

set_method("Classique")
app.mainloop()