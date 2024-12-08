import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from model import MultiDomainVAE
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from brisque import BRISQUE
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from tensorflow.keras.applications import VGG19

DATASET_DIR = "Dataset"
BATCH_SIZE = 16
IMAGE_SIZE = (256, 256)
LATENT_DIM = 1024
EPOCHS = 1
BETA = 0.001
MAX_BETA = 1.
ANNEAL_EPOCH = 3
TAUX_APPRENTISSAGE = 1e-6
ALPHA_MAX = 1.

PONDERATION_X_TRAIN = 1.
PONDERATION_Y_TRAIN = 1.
PONDERATION_Z_TRAIN = 1.

SHUFFLE = False

def get_feature_extractor(input_shape):
    vgg = VGG19(include_top=False, weights='imagenet', input_shape=input_shape)
    vgg.trainable = False
    feature_extractor = Model(inputs=vgg.input, outputs=vgg.get_layer("block3_conv3").output)
    return feature_extractor

def calculate_brisque(images, epoch, output_dir="Résultats"):
    """
    Calcule le score BRISQUE pour une liste d'images et sauvegarde les résultats dans un fichier texte.
    """
    os.makedirs(output_dir, exist_ok=True)
    scores = []

    for i, image in enumerate(images):

        img = (image.numpy().squeeze() * 255).astype("uint8")
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        brisque_evaluator = BRISQUE()
        score = brisque_evaluator.score(img)
        scores.append(score)
    
    with open(os.path.join(output_dir, f"brisque_scores_epoch{epoch}.txt"), "w") as f:
        for i, score in enumerate(scores):
            f.write(f"Image {i}: BRISQUE Score = {score:.2f}\n")

    avg_score = np.mean(scores)
    print(f"Score BRISQUE moyen pour l'époque {epoch}: {avg_score:.2f}")
    return avg_score

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
    
    Arguments :
    - epoch_losses : dict contenant les pertes 'reconstruction', 'kl' et 'total' sous forme de listes.
    - output_dir : str, chemin du répertoire où sauvegarder les graphiques.
    - epoch : int ou None, numéro d'époque à inclure dans le nom du fichier (optionnel).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    num_epochs = len(epoch_losses['reconstruction'])
    epochs_range = range(1, num_epochs + 1)
    xticks_labels = [str(epoch) if epoch == 1 or epoch % 5 == 0 else '' for epoch in epochs_range]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, epoch_losses['reconstruction'], label="Perte Reconstruction", marker='o')
    plt.plot(epochs_range, epoch_losses['kl'], label="Perte KL", marker='o')
    plt.plot(epochs_range, epoch_losses['ssim'], label="Perte SSIM", marker='o')
    plt.plot(epochs_range, epoch_losses['perceptual'], label="Perte Perceptuelle", marker='o')
    plt.plot(epochs_range, epoch_losses['total'], label="Perte Totale", marker='o')

    plt.xticks(epochs_range, xticks_labels, rotation=0)  # Rotation à 0 pour aligner les époques correctement
    plt.xlabel("Époque")
    plt.ylabel("Valeur de la Perte")
    plt.title("Évolution des Pertes")
    plt.legend()
    plt.grid()

    file_name = f"loss_curves_epoch{epoch}.png" if epoch is not None else "loss_curves.png"
    file_path = os.path.join(output_dir, file_name)

    plt.savefig(file_path)
    plt.close()
    print(f"Courbes des pertes sauvegardées dans : {file_path}")
    
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
def vae_loss(inputs, outputs, z_mean, z_log_var, feature_extractor=None, beta=1.0, alpha_max=1.0):
    """Calcule la perte de reconstruction, perceptuelle, SSIM et divergence KL."""
    
    # Perte de reconstruction (MSE ou BCE)
    reconstruction_loss = tf.reduce_mean(tf.square(inputs - outputs))  # MSE
    # reconstruction_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(inputs, outputs))  # BCE

    # Pondération adaptative pour la reconstruction
    alpha = tf.minimum(alpha_max, tf.exp(-beta))
    reconstruction_loss = alpha * reconstruction_loss

    # Perte perceptuelle
    perceptual_loss_value = 0
    if feature_extractor is not None:
        features_inputs = feature_extractor(inputs)
        features_outputs = feature_extractor(outputs)
        perceptual_loss_value = tf.reduce_mean(tf.abs(features_inputs - features_outputs))

    # Perte SSIM
    ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(inputs, outputs, max_val=1.0))

    # Perte KL (avec pondération beta)
    kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
    kl_loss = tf.reduce_mean(kl_loss)
    kl_loss = beta * kl_loss

    # Somme des pertes
    total_loss = reconstruction_loss + kl_loss + perceptual_loss_value + ssim_loss

    return reconstruction_loss, kl_loss, perceptual_loss_value, ssim_loss, total_loss
    
def train_step(model, inputs, optimizer, beta):
    """Effectue une étape d'entraînement."""
    with tf.GradientTape() as tape:
        outputs, latent_params = model(inputs, training=True)
        
        # Calcul des pertes pour chaque domaine
        loss_X = vae_loss(inputs['X'], outputs['X'], latent_params['z_mean_X'], latent_params['z_log_var_X'], beta=beta)
        loss_Y = vae_loss(inputs['Y'], outputs['Y'], latent_params['z_mean_Y'], latent_params['z_log_var_Y'], beta=beta)
        loss_Z = vae_loss(inputs['Z'], outputs['Z'], latent_params['z_mean_Z'], latent_params['z_log_var_Z'], beta=beta)
        
        # Pondération des pertes perceptuelles pour le calcul de total_loss
        reconstruction_loss = loss_X[0] + loss_Y[0] + loss_Z[0]
        kl_loss = loss_X[1] + loss_Y[1] + loss_Z[1]
        perceptual_loss = loss_X[2] + loss_Y[2] + loss_Z[2]
        ssim_loss = loss_X[3] + loss_Y[3] + loss_Z[3]

        # Assurez-vous que les pondérations sont des scalaires ou tenseurs compatibles TensorFlow
        total_loss = (
            tf.convert_to_tensor(PONDERATION_X_TRAIN, dtype=tf.float32) * loss_X[-1] +
            tf.convert_to_tensor(PONDERATION_Y_TRAIN, dtype=tf.float32) * loss_Y[-1] +
            tf.convert_to_tensor(PONDERATION_Z_TRAIN, dtype=tf.float32) * loss_Z[-1]
        )

    # Calcul des gradients et application
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Renvoie les différentes pertes pour le suivi
    return loss_X, loss_Y, loss_Z, total_loss

def train(model, datasets, epochs, optimizer, anneal_epochs=ANNEAL_EPOCH, max_beta=MAX_BETA):
    """Boucle d'entraînement principale avec suivi de PSNR, SSIM et BRISQUE."""
    epoch_losses = {'reconstruction': [], 'kl': [], 'ssim': [], 'perceptual': [], 'total': []}  # Suivi des pertes / époque
    epoch_metrics = {'psnr': [], 'ssim': [], 'brisque': []}  # PSNR, SSIM, BRISQUE par époque

    for epoch in range(epochs):
        print(f"Époque {epoch + 1}/{epochs}")
        epoch_loss = {'reconstruction': [], 'kl': [], 'ssim': [], 'perceptual': [], 'total': []}

        BETA = calculate_beta(epoch, max_beta=max_beta, anneal_epochs=anneal_epochs)  # MAJ de beta
        
        with tqdm(total=len(datasets['Real_Photos'])) as pbar:
            for batch_X, batch_Y, batch_Z in zip(datasets['Real_Photos'], datasets['Synthesized_Old_Photos'], datasets['Old_Photos']):
                # Préparez le batch comme dictionnaire
                inputs = {'X': batch_X, 'Y': batch_Y, 'Z': batch_Z}
                
                # Entraînement sur ce batch
                loss_X, loss_Y, loss_Z, total_loss = train_step(model, inputs, optimizer, beta=BETA)

                reconstruction_loss = tf.reduce_sum([loss_X[0], loss_Y[0], loss_Z[0]])
                kl_loss = tf.reduce_sum([loss_X[1], loss_Y[1], loss_Z[1]])
                perceptual_loss = tf.reduce_sum([loss_X[2], loss_Y[2], loss_Z[2]])
                ssim_loss = tf.reduce_sum([loss_X[3], loss_Y[3], loss_Z[3]])
                total_loss = tf.reduce_sum([loss_X[-1], loss_Y[-1], loss_Z[-1]])

                epoch_loss['reconstruction'].append(float(reconstruction_loss.numpy()))
                epoch_loss['kl'].append(float(kl_loss.numpy()))
                epoch_loss['perceptual'].append(float(perceptual_loss.numpy()))
                epoch_loss['ssim'].append(float(ssim_loss.numpy()))
                epoch_loss['total'].append(float(total_loss.numpy()))
                
                pbar.set_postfix({'Perte Moyenne': f"{np.mean(epoch_loss['total']):.6f}"})
                pbar.update(1)

        # Calcul des moyennes pour cette époque
        for key in epoch_losses.keys():
            epoch_losses[key].append(np.mean(epoch_loss[key]))

        # Évaluation des reconstructions pour PSNR, SSIM, et BRISQUE
        reconstructions = model(inputs, training=False)['X']
        psnr_scores = []
        ssim_scores = []
        brisque_scores = []

        for original, reconstructed in zip(inputs['X'], reconstructions):
            original_img = (original.numpy().squeeze() * 255).astype("uint8")
            reconstructed_img = (reconstructed.numpy().squeeze() * 255).astype("uint8")

            # Calculer PSNR et SSIM
            psnr_scores.append(psnr(original_img, reconstructed_img, data_range=255))
            ssim_scores.append(ssim(original_img, reconstructed_img, data_range=255))
            if len(reconstructed_img.shape) == 2:
                reconstructed_img = cv2.cvtColor(reconstructed_img, cv2.COLOR_GRAY2RGB)
            
            brisque_evaluator = BRISQUE()
            brisque_score = brisque_evaluator.score(reconstructed_img)
            brisque_scores.append(brisque_score)

        avg_psnr = np.mean(psnr_scores)
        avg_ssim = np.mean(ssim_scores)
        avg_brisque = np.mean(brisque_scores)

        # Enregistrer les métriques moyennes
        epoch_metrics['psnr'].append(avg_psnr)
        epoch_metrics['ssim'].append(avg_ssim)
        epoch_metrics['brisque'].append(avg_brisque)

        # Afficher les métriques moyennes
        print(f"PSNR moyen pour l'époque {epoch + 1}: {avg_psnr:.2f}")
        print(f"SSIM moyen pour l'époque {epoch + 1}: {avg_ssim:.2f}")
        print(f"BRISQUE moyen pour l'époque {epoch + 1}: {avg_brisque:.2f}")

        save_reconstructions(
            inputs=inputs['X'],  # Entrées du domaine X
            outputs=reconstructions,  # Reconstructions pour le domaine X
            epoch=epoch + 1
        )
        save_loss_curves(epoch_losses, epoch=epoch + 1)

    plot_metrics(epoch_metrics)
    print("Entraînement terminé.")

def plot_metrics(metrics, output_dir="Résultats"):
    """
    Traçage des courbes PSNR, SSIM et BRISQUE en fonction des époques.
    PSNR et BRISQUE sont tracés ensemble, tandis que SSIM est dans un graphique séparé.
    """
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(metrics['psnr']) + 1)

    xticks_labels = [str(epoch) if epoch % 5 == 0 else '' for epoch in epochs]

    # Graphique pour PSNR et BRISQUE
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics['psnr'], label="PSNR", marker='o')
    plt.plot(epochs, metrics['brisque'], label="BRISQUE", marker='o')
    plt.xticks(epochs, xticks_labels)
    plt.xlabel("Époque")
    plt.ylabel("Valeur des Métriques")
    plt.title("Évolution des Métriques (PSNR et BRISQUE)")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "metrics_psnr_brisque.png"))
    plt.close()

    # Graphique séparé pour SSIM
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics['ssim'], label="SSIM", marker='o', color='green')
    plt.xticks(epochs, xticks_labels)
    plt.xlabel("Époque")
    plt.ylabel("SSIM")
    plt.title("Évolution du SSIM")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "metrics_ssim.png"))
    plt.close()

if __name__ == "__main__":
    datasets = load_datasets(DATASET_DIR)
    datasets = align_datasets(datasets)
    
    model = MultiDomainVAE(latent_dim=LATENT_DIM, input_shape=(256, 256, 1))
    optimizer = Adam(learning_rate=TAUX_APPRENTISSAGE)
    
    train(model, datasets, epochs=EPOCHS, optimizer=optimizer)
    
    model.save("vae_full_model.keras")
    print("Poids finaux sauvegardés.")
