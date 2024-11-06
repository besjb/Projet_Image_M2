import cv2
import numpy as np

image = cv2.imread("Assets/Test.jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Impossible de charger l'image.")
    exit(-1)

dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shifted = np.fft.fftshift(dft)

magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shifted[:, :, 0], dft_shifted[:, :, 1]) + 1)

rows, cols = image.shape
crow, ccol = rows // 2 , cols // 2
radius = 30
mask = np.zeros((rows, cols, 2), np.uint8)
cv2.circle(mask, (ccol, crow), radius, (1, 1), -1)

# Application du filtre passe-bas
filtered_dft = dft_shifted * mask

magnitude_spectrum_filtered = 20 * np.log(cv2.magnitude(filtered_dft[:, :, 0], filtered_dft[:, :, 1]) + 1)

# Retour en espace image 
dft_inverse_shifted = np.fft.ifftshift(filtered_dft)  
filtered_image = cv2.idft(dft_inverse_shifted)
filtered_image = cv2.magnitude(filtered_image[:, :, 0], filtered_image[:, :, 1])

filtered_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX)
filtered_image = np.uint8(filtered_image)

cv2.imshow("Image originale", image)
cv2.imshow("Spectre des fréquences (avant filtrage)", np.uint8(magnitude_spectrum))
cv2.imshow("Spectre des fréquences (après filtrage)", np.uint8(magnitude_spectrum_filtered))
cv2.imshow("Image filtrée avec passe-bas", filtered_image)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cv2.destroyAllWindows()
