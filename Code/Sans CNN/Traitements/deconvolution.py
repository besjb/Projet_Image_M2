import cv2
import numpy as np

def wiener_deconvolution(image, kernel, noise_var, signal_var):
    # Appliquer la transformée de Fourier sur l'image et le noyau
    kernel_ft = np.fft.fft2(kernel, s=image.shape)  # Transformée du noyau
    image_ft = np.fft.fft2(image)  # Transformée de l'image
    kernel_ft_conj = np.conj(kernel_ft)  # Conjugué du noyau

    # Calcul de la dénomination (la partie du dénominateur de la formule de Wiener)
    denominator = np.abs(kernel_ft)**2 + (noise_var / signal_var)

    # Calcul du numérateur de la formule de Wiener
    numerator = kernel_ft_conj * image_ft

    # Appliquer la formule de la déconvolution de Wiener dans le domaine fréquentiel
    result_ft = numerator / denominator

    # Appliquer l'inverse de la transformée de Fourier pour obtenir l'image restaurée
    result = np.fft.ifft2(result_ft)
    return np.abs(result)

image = cv2.imread("Assets/inpainted_image.jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Impossible de charger l'image.")
    exit(-1)

kernel = np.ones((5, 5)) / 25

# Définir la variance du bruit (noise_var) et la variance du signal (signal_var)
noise_var = 0.1
signal_var = 1.0

# Déconvolution de Wiener
deconvolved_image = wiener_deconvolution(image, kernel, noise_var, signal_var)

cv2.imshow("Image Floue", image)
cv2.imshow("Image Deconvoluee", deconvolved_image.astype(np.uint8))

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
