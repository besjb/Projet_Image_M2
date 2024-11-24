import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import customtkinter as ctk
from tensorflow.keras.models import load_model
from restauration import process_images
from model import MultiDomainVAE

# Dossiers pour les images
INPUT_FOLDER = "Assets"
OUTPUT_FOLDER = "Restaurés"

class ImageRestorationApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Image Restauration")
        self.geometry("800x600")

        # Modèle sélectionné
        self.selected_model = None
        self.model = None  # Stocke le modèle chargé

        # Liste des fichiers restaurés
        self.restored_files = []
        self.current_index = 0

        # Interface
        self.create_widgets()

    def create_widgets(self):
        # Label pour le titre
        self.title_label = ctk.CTkLabel(self, text="Sélectionnez un modèle :", font=("Arial", 20))
        self.title_label.pack(pady=10)

        # Boutons pour choisir les modèles
        self.model_buttons_frame = ctk.CTkFrame(self)
        self.model_buttons_frame.pack(pady=10)

        for model_name in ["vae_full_model.keras", "test2", "test3"]:
            btn = ctk.CTkButton(self.model_buttons_frame, text=model_name, command=lambda m=model_name: self.select_model(m))
            btn.pack(side="left", padx=5)

        # Bouton pour lancer la restauration
        self.restore_button = ctk.CTkButton(self, text="Restaurer les images", command=self.run_restoration)
        self.restore_button.pack(pady=20)

        # Zone pour afficher les images
        self.image_frame = ctk.CTkFrame(self)
        self.image_frame.pack(pady=10, fill="both", expand=True)

        self.original_label = ctk.CTkLabel(self.image_frame, text="Image originale")
        self.original_label.grid(row=0, column=0, padx=10, pady=10)
        self.processed_label = ctk.CTkLabel(self.image_frame, text="Image restaurée")
        self.processed_label.grid(row=0, column=1, padx=10, pady=10)

        self.original_image = ctk.CTkLabel(self.image_frame, text="")
        self.original_image.grid(row=1, column=0, padx=10, pady=10)
        self.processed_image = ctk.CTkLabel(self.image_frame, text="")
        self.processed_image.grid(row=1, column=1, padx=10, pady=10)

        # Boutons de navigation
        self.navigation_frame = ctk.CTkFrame(self)
        self.navigation_frame.pack(pady=10)

        self.prev_button = ctk.CTkButton(self.navigation_frame, text="Précédent", command=self.show_previous_image)
        self.prev_button.pack(side="left", padx=5)

        self.next_button = ctk.CTkButton(self.navigation_frame, text="Suivant", command=self.show_next_image)
        self.next_button.pack(side="left", padx=5)

    def select_model(self, model_name):
        """Charge le modèle sélectionné."""
        #model_path = os.path.join("Models", model_name)
        try:
            self.model = load_model(model_name, custom_objects={"MultiDomainVAE": MultiDomainVAE})
            self.selected_model = model_name
            ctk.CTkLabel(self, text=f"Modèle chargé : {model_name}", font=("Arial", 14)).pack()
        except Exception as e:
            ctk.CTkLabel(self, text=f"Erreur : {str(e)}", font=("Arial", 14), fg_color="red").pack()

    def run_restoration(self):
        """Lance la restauration des images."""
        if not self.model:
            ctk.CTkLabel(self, text="Veuillez charger un modèle avant de restaurer.", font=("Arial", 14), fg_color="red").pack()
            return

        # Lance la restauration avec le script
        process_images(self.model, INPUT_FOLDER, OUTPUT_FOLDER)

        # Charge les fichiers restaurés
        self.restored_files = [f for f in os.listdir(OUTPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.current_index = 0
        if self.restored_files:
            self.show_image_pair()

    def show_image_pair(self):
        if not self.restored_files:
            return

        file_name = self.restored_files[self.current_index]
        original_path = os.path.join(INPUT_FOLDER, file_name)
        processed_path = os.path.join(OUTPUT_FOLDER, file_name)

        original_img = Image.open(original_path).resize((256, 256))
        processed_img = Image.open(processed_path).resize((256, 256))

        self.update_image(self.original_image, original_img)
        self.update_image(self.processed_image, processed_img)

    def update_image(self, label, img):
        img_tk = ImageTk.PhotoImage(img)
        label.configure(image=img_tk)
        label.image = img_tk

    def show_previous_image(self):
        if self.restored_files and self.current_index > 0:
            self.current_index -= 1
            self.show_image_pair()

    def show_next_image(self):
        if self.restored_files and self.current_index < len(self.restored_files) - 1:
            self.current_index += 1
            self.show_image_pair()

if __name__ == "__main__":
    app = ImageRestorationApp()
    app.mainloop()
