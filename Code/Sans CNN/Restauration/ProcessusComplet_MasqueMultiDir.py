import cv2
import numpy as np

# Déconvolution de Wiener (comme avant)
def wiener_deconvolution(image, kernel, noise_var, signal_var):
    kernel_ft = np.fft.fft2(kernel, s=image.shape)
    image_ft = np.fft.fft2(image)
    kernel_ft_conj = np.conj(kernel_ft)
    denominator = np.abs(kernel_ft)**2 + (noise_var / signal_var)
    numerator = kernel_ft_conj * image_ft
    result_ft = numerator / denominator
    result = np.fft.ifft2(result_ft)
    return np.abs(result)

# Estimation des variances de bruit et signal
def estimate_noise_and_signal_variance(image, filtered_image, kernel):
    signal_var = np.var(image)
    noise_estimate = image - filtered_image
    noise_var = np.var(noise_estimate)
    return noise_var, signal_var

# Expansion dynamique
def dynamic_range_expansion(image, y_min=0, y_max=255):
    x_min = np.min(image)
    x_max = np.max(image)
    alpha = (y_min * x_max - y_max * x_min) / (x_max - x_min)
    beta = (y_max - y_min) / (x_max - x_min)
    expanded_image = alpha + beta * image
    expanded_image = np.clip(expanded_image, y_min, y_max)
    return expanded_image.astype(np.uint8)

# Calcul du PSNR
def calculate_psnr(original, restored):
    return cv2.PSNR(original, restored)

# Calcul du SSIM
def calculate_ssim(original, restored):
    K1 = 0.01
    K2 = 0.03
    L = 255
    mu_x = np.mean(original)
    mu_y = np.mean(restored)
    sigma_x_sq = np.var(original)
    sigma_y_sq = np.var(restored)
    sigma_xy = np.cov(original.flatten(), restored.flatten())[0, 1]
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x_sq + sigma_y_sq + C2)
    ssim_value = numerator / denominator
    return ssim_value

# Distance de l'histogramme
def histogram_distance(image1, image2):
    hist1 = cv2.calcHist([image1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0, 256])
    hist1 /= hist1.sum()
    hist2 /= hist2.sum()
    hist_diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_KL_DIV)
    return hist_diff

# Préparation du masque automatiquement sans dessin manuel
def preprocess_image(image):
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    noyau = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    nettoye = cv2.filter2D(gris, -1, noyau)
    return nettoye

def filtres_passe_bande(image, nb_filtres=12):
    filtres = []
    pas_angle = 180 / nb_filtres
    for i in range(nb_filtres):
        angle = i * pas_angle
        noyau = cv2.getDerivKernels(1, 0, 1, 1, angle)
        noyau_dx = noyau[0]
        noyau_dy = noyau[1]
        noyau_combine = np.outer(noyau_dx, noyau_dy)
        filtres.append(noyau_combine)
    images_filtrees = [cv2.filter2D(image, -1, noyau) for noyau in filtres]
    return images_filtrees

def seuil_image(image_filtree):
    _, masque_binaire = cv2.threshold(image_filtree, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return masque_binaire

def detecter_lignes(masque_binaire):
    lignes = cv2.HoughLinesP(masque_binaire, 1, np.pi / 180, threshold=150, minLineLength=10, maxLineGap=5)
    return lignes

def dessiner_contours(image, lignes):
    masque = np.zeros_like(image)
    if lignes is not None:
        for ligne in lignes:
            x1, y1, x2, y2 = ligne[0]
            cv2.line(masque, (x1, y1), (x2, y2), (255), 2)
    return masque

def remplir_rayures(masque):
    taille_noyau = (5, 5)
    noyau = np.ones(taille_noyau, np.uint8)
    masque_rempli = cv2.morphologyEx(masque, cv2.MORPH_CLOSE, noyau)
    masque_dilate = cv2.dilate(masque_rempli, noyau, iterations=3)
    return masque_dilate

def restaurer_image(image):
    image_pretraitee = preprocess_image(image)
    images_filtrees = filtres_passe_bande(image_pretraitee)
    masques_binaires = [seuil_image(filt) for filt in images_filtrees]
    masque_combine = np.zeros_like(masques_binaires[0])
    for masque in masques_binaires:
        masque_combine = cv2.bitwise_or(masque_combine, masque)
    lignes = detecter_lignes(masque_combine)
    masque_rayure = dessiner_contours(image, lignes)
    masque_rempli = remplir_rayures(masque_rayure)
    return masque_rempli

# Application principale de l'inpainting
image = cv2.imread("Assets/Test10.jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Impossible de charger l'image.")
    exit(-1)

# Préparation du masque pour l'inpainting
masque_rempli = restaurer_image(cv2.imread("Assets/Test10.jpg"))

# S'assurer que le masque est binaire (0 et 255), uint8, et 1 canal
masque_rempli = np.uint8(masque_rempli > 0) * 255
if len(masque_rempli.shape) == 3:
    masque_rempli = cv2.cvtColor(masque_rempli, cv2.COLOR_BGR2GRAY)

# Appliquer l'inpainting avec le masque
inpainted_img = cv2.inpaint(image, masque_rempli, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

# Expansion dynamique et calcul des métriques
expanded_image = dynamic_range_expansion(inpainted_img)
psnr_value = calculate_psnr(image, inpainted_img)
ssim_value = calculate_ssim(image, inpainted_img)
hist_diff = histogram_distance(image, inpainted_img)

cv2.imshow("Inpainted Image", inpainted_img)
cv2.imshow("Masque", masque_rempli)

cv2.imwrite('Assets/inpainted_image.jpg', inpainted_img)
print(f"PSNR : {psnr_value} dB, SSIM : {ssim_value}, Histogramme : {hist_diff}")
cv2.waitKey(0)
