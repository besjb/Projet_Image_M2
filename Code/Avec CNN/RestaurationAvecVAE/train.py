import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from model import MultiDomainVAE
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

DATASET_DIR = "Dataset"
BATCH_SIZE = 32
IMAGE_SIZE = (256, 256)
LATENT_DIM = 256
EPOCHS = 2
BETA = 0.001
MAX_BETA = 1.
ANNEAL_EPOCH = 5
TAUX_APPRENTISSAGE = 1e-6
ALPHA_MAX = 10.

PONDERATION_X_TRAIN = 1.
PONDERATION_Y_TRAIN = 1.
PONDERATION_Z_TRAIN = 1.

SHUFFLE = False

def preprocess_image(image):
    """Prétraitement : normalisation et redimensionnement."""
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image

def save_reconstructions(inputs, outputs, epoch):
    for i in range(min(len(inputs), 5)): 
        original = inputs[i].numpy().squeeze()
        reconstructed = outputs[i].numpy().squeeze()
        plt.imsave(f"Résultats/original_epoch{epoch}_sample{i}.png", original, cmap='gray')
        plt.imsave(f"Résultats/reconstructed_epoch{epoch}_sample{i}.png", reconstructed, cmap='gray')

def save_loss_curves(epoch_losses, output_dir="Résultats", epoch=None):
    """
    Sauvegarde les courbes des pertes (reconstruction, KL, totale) dans un fichier PNG.
    """
    os.makedirs(output_dir, exist_ok=True)
    num_epochs = len(epoch_losses['reconstruction']) 
    epochs_range = range(1, num_epochs + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, epoch_losses['reconstruction'], label="Perte Reconstruction", marker='o')
    plt.plot(epochs_range, epoch_losses['kl'], label="Perte KL", marker='o')
    plt.plot(epochs_range, epoch_losses['total'], label="Perte Totale", marker='o')
    plt.xticks(epochs_range)
    plt.xlabel("Époque")
    plt.ylabel("Valeur de la Perte")
    plt.title("Évolution des Pertes")
    plt.legend()
    plt.grid()
    file_name = f"loss_curves_epoch{epoch}.png" if epoch is not None else "loss_curves.png"
    plt.savefig(os.path.join(output_dir, file_name))
    plt.close()

def align_datasets(datasets):
    """Aligne les datasets sur la longueur minimale commune."""
    min_length = min(len(datasets['Real_Photos']), len(datasets['Synthesized_Old_Photos']), len(datasets['Old_Photos']))
    aligned_datasets = {
        key: datasets[key].take(min_length)
        for key in datasets
    }
    return aligned_datasets

def load_datasets(base_path):
    """Charge les datasets pour les domaines R, X, Z."""
    datasets = {}
    for domain in ['Real_Photos', 'Synthesized_Old_Photos', 'Old_Photos']:
        dataset_path = os.path.join(base_path, domain)
        dataset = tf.keras.preprocessing.image_dataset_from_directory(
            dataset_path,
            label_mode=None,
            color_mode='grayscale',
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            shuffle=SHUFFLE
        ).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        
        datasets[domain] = dataset.map(preprocess_image)
    return datasets

def calculate_beta(epoch, max_beta=MAX_BETA, anneal_epochs=ANNEAL_EPOCH, method='linear'):
    """
    MAJ de Beta selon différentes méthodes.
    """
    if method == 'linear':
        if epoch < anneal_epochs:
            return max_beta * (epoch / anneal_epochs)
        else:
            return max_beta
    elif method == 'exponential':
        if epoch < anneal_epochs:
            return max_beta * (1 - np.exp(-epoch / anneal_epochs))
        else:
            return max_beta
    elif method == 'cosine':
        if epoch < anneal_epochs:
            return max_beta * (1 - np.cos(np.pi * epoch / (2 * anneal_epochs)))
        else:
            return max_beta
    elif method == 'sigmoid':
        if epoch < anneal_epochs:
            return max_beta / (1 + np.exp(-10 * (epoch / anneal_epochs - 0.5)))
        else:
            return max_beta
    elif method == 'cyclic':
        cycle_length = anneal_epochs
        cycle_position = epoch % cycle_length
        return max_beta * (1 - np.cos(np.pi * cycle_position / cycle_length)) / 2
    elif method == 'stepwise':
        step_epochs = anneal_epochs // 5  # Divisez en 5 étapes
        return min(max_beta, max_beta * (epoch // step_epochs) / (10 // step_epochs))
    else:
        raise ValueError(f"Unknown method: {method}")

# Fonction de perte
def vae_loss(inputs, outputs, z_mean, z_log_var, beta=1.0):
    """Calcule la perte de reconstruction et la divergence KL."""

    # la cross entropie peut être trop sévère pour des données d'intensités continue comme les images / utiliser MSE
    #reconstruction_loss = tf.reduce_mean(tf.square(inputs - outputs)) # MSE
    reconstruction_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(inputs, outputs)) # ENtropie croisée binaire
    alpha = tf.minimum(ALPHA_MAX, 1.0 / (beta + 1e-6))
    reconstruction_loss = alpha * reconstruction_loss
    
    kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
    kl_loss = tf.reduce_mean(kl_loss)

    # La perte KL et la perte totale peuvent ne pas converger efficacement -> utilisation de Beta
    kl_loss = beta * kl_loss
    
    return reconstruction_loss, kl_loss, reconstruction_loss + kl_loss

def train_step(model, inputs, optimizer, beta):
    """Effectue une étape d'entraînement."""
    with tf.GradientTape() as tape:
        outputs, latent_params = model(inputs, training=True)

        # Pertes de reconstruction et KL
        loss_X = vae_loss(inputs['X'], outputs['X'], latent_params[0], latent_params[1], beta=beta)
        loss_Y = vae_loss(inputs['Y'], outputs['Y'], latent_params[2], latent_params[3], beta=beta)
        loss_Z = vae_loss(inputs['Z'], outputs['Z'], latent_params[4], latent_params[5], beta=beta)

        # Vérifiez les formes avant le calcul des pertes de consistance
        if latent_params[4].shape[0] != outputs['z_Z_mapped'].shape[0]:
            min_batch_size = min(latent_params[4].shape[0], outputs['z_Z_mapped'].shape[0])
            latent_params[4] = latent_params[4][:min_batch_size]
            outputs['z_Z_mapped'] = outputs['z_Z_mapped'][:min_batch_size]

        # Perte de consistance inter-domaines
        consistency_loss_Y = tf.reduce_mean(tf.square(latent_params[2] - outputs['z_Y_mapped']))
        consistency_loss_Z = tf.reduce_mean(tf.square(latent_params[4] - outputs['z_Z_mapped']))

        # Total des pertes
        total_loss = (
            loss_X[2] + loss_Y[2] + loss_Z[2] +
            consistency_loss_Y + consistency_loss_Z
        )
    
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss_X, loss_Y, loss_Z, total_loss

def train(model, datasets, epochs, optimizer, anneal_epochs=ANNEAL_EPOCH, max_beta=MAX_BETA):
    """Boucle d'entraînement principale avec barre de progression."""
    epoch_losses = {'reconstruction': [], 'kl': [], 'total': []}  # Suivi des pertes / époque

    for epoch in range(epochs):
        print(f"Époque {epoch + 1}/{epochs}")
        epoch_loss = {'reconstruction': [], 'kl': [], 'total': []}

        BETA = calculate_beta(epoch, max_beta=max_beta, anneal_epochs=anneal_epochs)  # MAJ de beta
        
        with tqdm(total=len(datasets['Real_Photos'])) as pbar:
            for batch_X, batch_Y, batch_Z in zip(datasets['Real_Photos'], datasets['Synthesized_Old_Photos'], datasets['Old_Photos']):

                # Préparez le batch comme dictionnaire
                inputs = {'X': batch_X, 'Y': batch_Y, 'Z': batch_Z}
                
                # Entraînement sur ce batch
                loss_X, loss_Y, loss_Z, total_loss = train_step(model, inputs, optimizer, beta=BETA)
                
                epoch_loss['reconstruction'].append(loss_X[0].numpy() + loss_Y[0].numpy() + loss_Z[0].numpy())
                epoch_loss['kl'].append(loss_X[1].numpy() + loss_Y[1].numpy() + loss_Z[1].numpy())
                epoch_loss['total'].append(total_loss.numpy())
                
                pbar.set_postfix({'Perte Moyenne': f"{np.mean(epoch_loss['total']):.6f}"})
                pbar.update(1)
        

        # Calcul des moyennes pour cette époque
        epoch_losses['reconstruction'].append(np.mean(epoch_loss['reconstruction']))
        epoch_losses['kl'].append(np.mean(epoch_loss['kl']))
        epoch_losses['total'].append(np.mean(epoch_loss['total']))

        # Métriques
        print(f"Perte Moyenne de l'Époque {epoch + 1}: {np.mean(epoch_loss['total']):.6f}")

        save_reconstructions(
            inputs=inputs['X'],  # Entrées du domaine X
            outputs=model(inputs, training=False)['X'],  # Reconstructions pour le domaine X
            epoch=epoch + 1
        )
        save_loss_curves(epoch_losses, epoch=epoch + 1)

    print("Entraînement terminé.")

if __name__ == "__main__":
    datasets = load_datasets(DATASET_DIR)
    datasets = align_datasets(datasets)
    
    model = MultiDomainVAE(latent_dim=LATENT_DIM, input_shape=(256, 256, 1))
    optimizer = Adam(learning_rate=TAUX_APPRENTISSAGE)
    
    train(model, datasets, epochs=EPOCHS, optimizer=optimizer)
    
    model.save("vae_full_model.keras")
    print("Poids finaux sauvegardés.")
