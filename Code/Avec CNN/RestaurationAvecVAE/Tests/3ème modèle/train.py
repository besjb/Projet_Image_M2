import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.preprocessing import image_dataset_from_directory
from PIL import Image
import logging
import csv
from model import MultiDomainVAE 

def validate_and_clean_images(directory):
    """Valide les fichiers d'image et supprime ceux qui sont invalides."""
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if not any(file.lower().endswith(ext) for ext in valid_extensions):
                print(f"Invalid file found and removed: {file_path}")
                os.remove(file_path)
                continue
            try:
                with Image.open(file_path) as img:
                    img.verify()
            except Exception as e:
                print(f"Corrupted image found and removed: {file_path}, Error: {e}")
                os.remove(file_path)

def custom_vae_loss(inputs, outputs, z_mean, z_log_var):
    print(f"Inputs shape: {inputs.shape}, Outputs shape: {outputs.shape}")
    
    reconstruction_loss = tf.reduce_mean(binary_crossentropy(
        tf.keras.backend.flatten(inputs), tf.keras.backend.flatten(outputs)
    ))
    kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
    total_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
    return reconstruction_loss, kl_loss, total_loss


def train_step(model, inputs, optimizer):
    with tf.GradientTape() as tape:
        outputs = model(inputs, training=True)
        recon_loss, kl_loss, total_loss = custom_vae_loss(
            inputs['X'], outputs['X'], model.encoder_X(inputs['X'])[0], model.encoder_X(inputs['X'])[1]
        )
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return recon_loss, kl_loss, total_loss

def load_dataset_from_directory(data_dir, batch_size=1, img_height=256, img_width=256):
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
    return dataset.map(lambda x: tf.cast(x, tf.float32) / 255.0)

def train(model, datasets, optimizer, epochs):
    logging.basicConfig(level=logging.INFO)
    best_loss = float('inf')

    metrics_file = "training_metrics.csv"
    with open(metrics_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Average Total Loss", "Average Reconstruction Loss", "Average KL Loss"])

    for epoch in range(epochs):
        epoch_total_losses = []
        epoch_recon_losses = []
        epoch_kl_losses = []

        for batch, (x, r, y) in enumerate(tf.data.Dataset.zip((datasets['X'], datasets['R'], datasets['Y']))):
            inputs = {'X': x, 'R': r, 'Y': y}
            recon_loss, kl_loss, total_loss = train_step(model, inputs, optimizer)
            
            epoch_total_losses.append(total_loss.numpy())
            epoch_recon_losses.append(recon_loss.numpy())
            epoch_kl_losses.append(kl_loss.numpy())

            logging.info(f"Epoch {epoch+1}, Batch {batch+1}, Total Loss: {total_loss.numpy()}, "
                         f"Reconstruction Loss: {recon_loss.numpy()}, KL Divergence: {kl_loss.numpy()}")

        avg_epoch_total_loss = np.mean(epoch_total_losses)
        avg_epoch_recon_loss = np.mean(epoch_recon_losses)
        avg_epoch_kl_loss = np.mean(epoch_kl_losses)

        with open(metrics_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, avg_epoch_total_loss, avg_epoch_recon_loss, avg_epoch_kl_loss])

        logging.info(f"Epoch {epoch+1} completed, Average Total Loss: {avg_epoch_total_loss}, "
                     f"Average Reconstruction Loss: {avg_epoch_recon_loss}, Average KL Loss: {avg_epoch_kl_loss}")

        if avg_epoch_total_loss < best_loss:
            best_loss = avg_epoch_total_loss
            model.save_weights('best_model_weights.weights.h5')
            logging.info(f"New best loss {best_loss}. Saved model weights.")

if __name__ == "__main__":
    model = MultiDomainVAE(latent_dim=256)
    optimizer = Adam()
    model.compile(optimizer=optimizer)

    dataset_dir = 'Dataset'
    datasets = {
        'X': preprocess_images(load_dataset_from_directory(f'{dataset_dir}/Synthesized_Old_Photos')),
        'R': preprocess_images(load_dataset_from_directory(f'{dataset_dir}/Real_Photos')),
        'Y': preprocess_images(load_dataset_from_directory(f'{dataset_dir}/Old_Photos'))
    }

    train(model, datasets, optimizer, epochs=1)
    print(f"Metrics recorded in training_metrics.csv")