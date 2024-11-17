import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, Adamax, SGD
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.preprocessing import image_dataset_from_directory
from PIL import Image
import logging
import csv

from model import MultiDomainVAE

def validate_and_clean_images(directory):
    """Valide et supprime les fichiers d'image invalides."""
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if not any(file.lower().endswith(ext) for ext in valid_extensions):
                print(f"Fichier invalide trouvé et supprimé : {file_path}")
                os.remove(file_path)
                continue
            try:
                with Image.open(file_path) as img:
                    img.verify()
            except Exception as e:
                print(f"Image corrompue trouvée et supprimée : {file_path}, Erreur : {e}")
                os.remove(file_path)

def custom_vae_loss(inputs, outputs, z_mean, z_log_var):
    """
    Fonction de perte personnalisée combinant la perte de reconstruction et la divergence KL.
    Ajuste les dimensions si nécessaire pour garantir la compatibilité.
    """
    outputs = tf.image.resize(outputs, tf.shape(inputs)[1:3])  # Redimensionne les sorties

    reconstruction_loss = tf.reduce_mean(
        binary_crossentropy(
            tf.keras.backend.flatten(inputs),
            tf.keras.backend.flatten(outputs)
        )
    )

    kl_loss = -0.5 * tf.reduce_sum(
        1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
        axis=-1
    )
    kl_loss = tf.reduce_mean(kl_loss)

    total_loss = reconstruction_loss + kl_loss
    return reconstruction_loss, kl_loss, total_loss

def train_step(model, inputs, optimizer):
    """Effectue une étape d'entraînement."""
    with tf.GradientTape() as tape:
        outputs = model(inputs, training=True)
        recon_loss, kl_loss, total_loss = custom_vae_loss(
            inputs['X'], outputs['X'],
            model.encoder_X(inputs['X'])[0], model.encoder_X(inputs['X'])[1]
        )
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return recon_loss, kl_loss, total_loss

def load_dataset_from_directory(data_dir, batch_size=50, img_height=256, img_width=256):
    """
    Charge les images sous forme de dataset TensorFlow et les prétraite.
    """
    validate_and_clean_images(data_dir)
    return image_dataset_from_directory(
        data_dir,
        label_mode=None,
        color_mode='rgb',
        image_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=True
    )

def preprocess_images(dataset):
    """Normalise les images dans l'intervalle [0, 1]."""
    return dataset.map(lambda x: tf.cast(x, tf.float32) / 255.0)

def train(model, datasets, optimizer, epochs):
    """Entraîne le modèle VAE avec les datasets donnés."""
    logging.basicConfig(level=logging.INFO)
    best_loss = float('inf')

    # Création ou écrasement du fichier CSV pour stocker les métriques
    metrics_file = "training_metrics.csv"
    with open(metrics_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Résumé des époques"])

    for epoch in range(epochs):
        epoch_total_losses = []
        epoch_recon_losses = []
        epoch_kl_losses = []

        for batch, (x, r, y) in enumerate(tf.data.Dataset.zip((datasets['X'], datasets['R'], datasets['Y']))):
            inputs = {'X': x, 'R': r, 'Y': y}
            recon_loss, kl_loss, total_loss = train_step(model, inputs, optimizer)

            # Collecte des pertes pour les moyennes
            epoch_total_losses.append(total_loss.numpy())
            epoch_recon_losses.append(recon_loss.numpy())
            epoch_kl_losses.append(kl_loss.numpy())

            logging.info(f"Époque : {epoch + 1}, Lot : {batch + 1}, "
                         f"Perte totale : {total_loss.numpy():.6f}, "
                         f"Perte de reconstruction : {recon_loss.numpy():.6f}, "
                         f"Perte KL : {kl_loss.numpy():.6f}")

        # Calcul des moyennes
        avg_total_loss = np.mean(epoch_total_losses)
        avg_recon_loss = np.mean(epoch_recon_losses)
        avg_kl_loss = np.mean(epoch_kl_losses)

        with open(metrics_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            formatted_line = (
                    f"Époque : {epoch + 1} / "
                    f"Perte totale moyenne : {avg_total_loss:.6f} / "
                    f"Perte de reconstruction moyenne : {avg_recon_loss:.6f} / "
                    f"Perte KL moyenne : {avg_kl_loss:.6f}"
                )
            writer.writerow([formatted_line])

        logging.info(f"Époque : {epoch + 1}, "
                     f"Perte totale moyenne : {avg_total_loss:.6f}, "
                     f"Perte de reconstruction moyenne : {avg_recon_loss:.6f}, "
                     f"Perte KL moyenne : {avg_kl_loss:.6f}")

        if avg_total_loss < best_loss:
            best_loss = avg_total_loss
            model.save_weights('best_model_weights.weights.h5')
            logging.info(f"Nouvelle meilleure perte {best_loss:.6f}. Poids du modèle sauvegardés.")


if __name__ == "__main__":
    model = MultiDomainVAE(latent_dim=256)
    optimizer = SGD()
    model.compile(optimizer=optimizer)

    dataset_dir = 'Dataset'

    datasets = {
        'X': preprocess_images(load_dataset_from_directory(f'{dataset_dir}/Synthesized_Old_Photos')),
        'R': preprocess_images(load_dataset_from_directory(f'{dataset_dir}/Real_Photos')),
        'Y': preprocess_images(load_dataset_from_directory(f'{dataset_dir}/Old_Photos'))
    }

    train(model, datasets, optimizer, epochs=30)
    print("Entraînement terminé. Métriques enregistrées dans training_metrics.csv.")
