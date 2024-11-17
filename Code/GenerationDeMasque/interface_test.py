import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from vae_model import VAE

# Charger le modèle VAE
vae = VAE(latent_dim=256)
vae.load_weights("vae_weights_256x256_segmentation.weights.h5")

def generate_mask(image_path):
    img = load_img(image_path, target_size=(256, 256), color_mode="grayscale")
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    mask, _, _ = vae(img_array)
    
    mask_img = (mask.numpy().squeeze() * 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask_img) 
    return mask_pil


class ImageRestorationApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Image Restoration")
        self.geometry("1200x700")
        self.configure(fg_color="white")

        # Dimensions des Canvas
        self.canvas_width = 500
        self.canvas_height = 500

        # Variables
        self.original_image = None
        self.restored_image = None
        self.image_path = None

        # Widgets
        self.create_widgets()

    def create_widgets(self):
        # Cadre principal
        self.frame = ctk.CTkFrame(self, corner_radius=10)
        self.frame.pack(pady=20, padx=20, fill="both", expand=True)

        # Zone d'affichage des images (côte à côte)
        self.image_frame = ctk.CTkFrame(self.frame)
        self.image_frame.pack(pady=10, fill="both", expand=True)

        self.canvas_in = tk.Canvas(
            self.image_frame, bg="white", width=self.canvas_width, height=self.canvas_height
        )
        self.canvas_in.pack(side="left", padx=10, pady=10)

        self.canvas_out = tk.Canvas(
            self.image_frame, bg="white", width=self.canvas_width, height=self.canvas_height
        )
        self.canvas_out.pack(side="left", padx=10, pady=10)

        # Boutons en bas
        self.button_frame = ctk.CTkFrame(self.frame)
        self.button_frame.pack(pady=10, side="bottom", fill="x")

        self.open_button = ctk.CTkButton(self.button_frame, text="Ouvrir Image", command=self.open_image)
        self.open_button.pack(side="left", padx=10)

        self.restore_button = ctk.CTkButton(
            self.button_frame, text="Générer Image Restaurée", command=self.restore_image, state="disabled"
        )
        self.restore_button.pack(side="left", padx=10)

        self.save_button = ctk.CTkButton(self.button_frame, text="Enregistrer Image Restaurée", command=self.save_image, state="disabled")
        self.save_button.pack(side="left", padx=10)

    def open_image(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if self.file_path:
            # Charger l'image en niveaux de gris
            self.original_image = Image.open(self.file_path).convert("L")
            self.display_image(self.original_image, self.canvas_in)
            self.restore_button.configure(state="normal")

    def restore_image(self):
        if self.original_image is not None:
            self.restored_image = generate_mask(self.file_path)
            self.display_image(self.restored_image, self.canvas_out)
            self.save_button.configure(state="normal")

    def save_image(self):
        if self.restored_image is not None:
            file_path_save = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
            if file_path_save:
                self.restored_image.save(file_path_save)

    def display_image(self, image, canvas):
        # Redimensionner l'image pour tenir dans le Canvas
        img_width, img_height = image.size
        scale = min(self.canvas_width / img_width, self.canvas_height / img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)

        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        image_tk = ImageTk.PhotoImage(resized_image)

        # Centrer l'image dans le Canvas
        x_offset = (self.canvas_width - new_width) // 2
        y_offset = (self.canvas_height - new_height) // 2
        canvas.delete("all")
        canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=image_tk)
        canvas.image = image_tk  # Sauvegarde pour éviter le garbage collection

if __name__ == "__main__":
    app = ImageRestorationApp()
    app.mainloop()
