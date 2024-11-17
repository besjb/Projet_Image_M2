import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import optimizers  # Import optimizers directly
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  # Ensure callbacks are imported
from vae_model import VAE

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            img = load_img(os.path.join(folder, filename), target_size=(256, 256), color_mode="grayscale")
            images.append(img_to_array(img) / 255.0)
    return np.array(images)

# Load your datasets
noisy_images = load_images_from_folder("Dataset_preprocessed/noisy_images")
degraded_images = load_images_from_folder("Dataset_preprocessed/degraded_images")

# Initialize the VAE model
vae = VAE(latent_dim=256)
vae.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss="mse")

# Setup callbacks for early stopping and saving the model
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(
    'best_vae_weights.weights.h5',  # Changed filename to end with `.weights.h5`
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=True
)


# Train the model with degraded images as input and restored images as targets
history = vae.fit(
    noisy_images, 
    degraded_images, 
    batch_size=128, 
    epochs=150, 
    validation_split=0.4,
    callbacks=[early_stopping, model_checkpoint]
)

vae.save_weights("vae_weights_256x256.weights.h5")
