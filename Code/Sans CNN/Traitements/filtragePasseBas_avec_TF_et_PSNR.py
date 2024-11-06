import cv2
import numpy as np

def calculate_psnr(original, processed):
    mse = np.mean((original - processed) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    return psnr

image = cv2.imread("Assets/Test2.jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Impossible de charger l'image.")
    exit(-1)

height, width = image.shape[:2]
print(f"Résolution de l'image : {width} x {height} pixels")

best_psnr = 0
best_filtered_image = None
best_magnitude_spectrum_filtered = None
best_radius = 0

dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shifted = np.fft.fftshift(dft)

magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shifted[:, :, 0], dft_shifted[:, :, 1]) + 1)

rows, cols = image.shape
crow, ccol = rows // 2, cols // 2

for radius in range(10, int(min(height/2, width/2)), 10):  # Tester des rayons par interval de 10
    mask = np.zeros((rows, cols, 2), np.uint8)
    cv2.circle(mask, (ccol, crow), radius, (1, 1), -1)
    
    # Application du filtre passe-bas
    filtered_dft = dft_shifted * mask
    magnitude_spectrum_filtered = 20 * np.log(cv2.magnitude(filtered_dft[:, :, 0], filtered_dft[:, :, 1]) + 1)
    
    # Transformation de Fourier inverse pour revenir à l'espace image
    dft_inverse_shifted = np.fft.ifftshift(filtered_dft)
    filtered_image = cv2.idft(dft_inverse_shifted)
    filtered_image = cv2.magnitude(filtered_image[:, :, 0], filtered_image[:, :, 1])
    
    filtered_image_normalized = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX)
    filtered_image_normalized = np.uint8(filtered_image_normalized)
    
    # Calculer le PSNR entre l'image originale et l'image filtrée
    psnr = calculate_psnr(image, filtered_image_normalized)
    print(f"Rayon: {radius}, PSNR: {psnr:.2f} dB")
    
    if psnr > best_psnr:
        best_psnr = psnr
        best_filtered_image = filtered_image_normalized
        best_magnitude_spectrum_filtered = magnitude_spectrum_filtered
        best_radius = radius

print(f"Meilleur rayon: {best_radius} avec un PSNR de {best_psnr:.2f} dB")

cv2.imshow("Image originale", image)
cv2.imshow("Spectre des fréquences (avant filtrage)", np.uint8(magnitude_spectrum))
cv2.imshow("Spectre des fréquences (meilleur filtrage)", np.uint8(best_magnitude_spectrum_filtered))
cv2.imshow("Image filtrée avec meilleur PSNR", best_filtered_image)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
