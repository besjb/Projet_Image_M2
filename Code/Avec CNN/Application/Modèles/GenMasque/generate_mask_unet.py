from unet_model import unet
from data import testGenerator, saveResult
import os

# Chargement du modèle
model = unet(pretrained_weights="unet_trained_model.keras")

# Génération des masques
test_path = "Assets"
output_path = "Assets_masks"
test_gen = testGenerator(test_path, target_size=(256, 256))

# Prédictions
results = model.predict(test_gen, verbose=1)

# Sauvegarde des résultats
if not os.path.exists(output_path):
    os.makedirs(output_path)

saveResult(output_path, results, threshold=0.25)