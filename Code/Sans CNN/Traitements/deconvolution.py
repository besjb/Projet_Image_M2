import cv2
import numpy as np

def wiener_deconvolution(image, kernel, noise_var, signal_var):
    kernel_ft = np.fft.fft2(kernel, s=image.shape)
    image_ft = np.fft.fft2(image)
    kernel_ft_conj = np.conj(kernel_ft)

    denominator = np.abs(kernel_ft)**2 + noise_var / signal_var
    numerator = kernel_ft_conj * image_ft

    result_ft = numerator / denominator
    result = np.fft.ifft2(result_ft)
    return np.abs(result)

image = cv2.imread("Assets/Test7.png", cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Impossible de charger l'image.")
    exit(-1)

kernel = np.ones((5, 5)) / 25

noise_var = 0.1  
signal_var = 1.0  

deconvolved_image = wiener_deconvolution(image, kernel, noise_var, signal_var)

cv2.imshow("Image Floue", image)
cv2.imshow("Image Déconvoluée", deconvolved_image.astype(np.uint8))

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cv2.destroyAllWindows()
