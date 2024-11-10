import cv2
import numpy as np
from scipy.ndimage import convolve

# Chemin du fichier XML pour la détection de visages
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# Déconvolution de Wiener
def wiener_deconvolution(image, kernel, noise_var, signal_var):
    kernel_ft = np.fft.fft2(kernel, s=image.shape)
    image_ft = np.fft.fft2(image)
    kernel_ft_conj = np.conj(kernel_ft)
    denominator = np.abs(kernel_ft)**2 + (noise_var / signal_var)
    numerator = kernel_ft_conj * image_ft
    result_ft = numerator / denominator
    result = np.fft.ifft2(result_ft)
    return np.abs(result)

# Expansion dynamique
def dynamic_range_expansion(image, y_min=0, y_max=255):
    x_min = np.min(image)
    x_max = np.max(image)
    alpha = (y_min * x_max - y_max * x_min) / (x_max - x_min)
    beta = (y_max - y_min) / (x_max - x_min)
    expanded_image = alpha + beta * image
    expanded_image = np.clip(expanded_image, y_min, y_max)
    return expanded_image.astype(np.uint8)

# Calcul du PSNR et SSIM
def calculate_psnr(original, restored):
    return cv2.PSNR(original, restored)

def calculate_ssim(original, restored):
    K1, K2, L = 0.01, 0.03, 255
    mu_x, mu_y = np.mean(original), np.mean(restored)
    sigma_x_sq, sigma_y_sq = np.var(original), np.var(restored)
    sigma_xy = np.cov(original.flatten(), restored.flatten())[0, 1]
    C1, C2 = (K1 * L) ** 2, (K2 * L) ** 2
    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x_sq + sigma_y_sq + C2)
    return numerator / denominator

# Détection de rayures par magnitude du gradient
def gradient_magnitude(image):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    grad_x = convolve(image, sobel_x)
    grad_y = convolve(image, sobel_y)
    return np.sqrt(grad_x**2 + grad_y**2)

# Création automatique du masque de rayures
def create_scratch_mask(image):
    gSize = 35
    gaussian_filtered = cv2.GaussianBlur(image, (gSize, gSize), gSize / 2)
    gradient_magnitude_image = gradient_magnitude(gaussian_filtered)
    threshold_value = 0.3 * np.max(gradient_magnitude_image)
    binary_image = gradient_magnitude_image > threshold_value
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed_image = cv2.morphologyEx(binary_image.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed_image)
    filtered_image = np.zeros_like(closed_image)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 500:
            filtered_image[labels == i] = 255
    return cv2.morphologyEx(filtered_image, cv2.MORPH_CLOSE, kernel)

# Détection des visages et création d'un masque visage
def create_face_mask(image):
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=6, minSize=(30, 30))
    face_mask = np.zeros_like(image, dtype=np.uint8)
    for (x, y, w, h) in faces:
        cv2.rectangle(face_mask, (x, y), (x + w, y + h), 255, -1)
    return face_mask

image_path = 'Assets/Test10.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Erreur lors du chargement de l'image.")
    exit(-1)

scratch_mask = create_scratch_mask(image)
face_mask = create_face_mask(image)
scratch_mask = cv2.bitwise_and(scratch_mask, cv2.bitwise_not(face_mask))

# Inpainting et restauration de l'image
inpainted_img = cv2.inpaint(image, scratch_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

# Expansion dynamique
expanded_image = dynamic_range_expansion(inpainted_img)
cv2.imshow("Expansion dynamique de l'image (apres deconvolution)", expanded_image)

psnr_value = calculate_psnr(image, inpainted_img)
ssim_value = calculate_ssim(image, inpainted_img)

print(f"PSNR de l'image restaurée : {psnr_value} dB")
print(f"SSIM de l'image restaurée : {ssim_value}")

cv2.imshow('Original Image', image)
cv2.imshow('Inpainted Image', inpainted_img)
cv2.imshow('Detected Scratch Mask (Inverted)', scratch_mask)
cv2.imshow('Face Mask', face_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
