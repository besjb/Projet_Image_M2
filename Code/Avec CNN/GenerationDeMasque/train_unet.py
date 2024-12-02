from unet_model import unet
from data import trainGenerator, saveResult
import matplotlib.pyplot as plt


# Paramètres d'entraînement
train_path = "Dataset_preprocessed"
image_folder = "degraded_images"
mask_folder = "masks"
batch_size = 4
epochs = 50
steps_per_epoch = 300
target_size = (256, 256)

# Générateur d'images
data_gen_args = dict(rotation_range=0.2,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     shear_range=0.05,
                     zoom_range=0.05,
                     horizontal_flip=True,
                     fill_mode='nearest')
train_gen = trainGenerator(batch_size, train_path, image_folder, mask_folder, data_gen_args, target_size=target_size)

# Initialisation et entraînement du modèle
model = unet()
history = model.fit(train_gen, steps_per_epoch=steps_per_epoch, epochs=epochs)

# Sauvegarde des poids du modèle
model.save("unet_trained_model.keras")

# Afficher la courbe de l'évolution du loss
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Train Loss')

if 'val_loss' in history.history:
    plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Évolution de la perte au cours des époques')
plt.xlabel('Époques')
plt.ylabel('Perte')
plt.legend()
plt.grid()
plt.show()

if 'accuracy' in history.history:  # Vérifiez si 'accuracy' est calculée
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Évolution de l’accuracy au cours des époques')
    plt.xlabel('Époques')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.show()
