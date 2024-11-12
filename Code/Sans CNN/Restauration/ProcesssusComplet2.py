import cv2
import numpy as np

# Déconvolution de Wiener
def wiener_deconvolution(image, kernel, noise_var, signal_var):
    kernel_ft = np.fft.fft2(kernel, s=image.shape)
    image_ft = np.fft.fft2(image)
    kernel_ft_conj = np.conj(kernel_ft) 

    # Ajout d'un petit epsilon pour éviter les divisions par des valeurs très faibles
    epsilon = 1e-6
    denominator = np.abs(kernel_ft)**2 + (noise_var / signal_var) + epsilon
    numerator = kernel_ft_conj * image_ft
    result_ft = numerator / denominator

    result = np.fft.ifft2(result_ft)
    result = np.abs(result)
    result = np.clip(result, 0, 255)
    return result.astype(np.uint8)

# Estimation des variances de bruit et signal
def estimate_noise_and_signal_variance(image, filtered_image):
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
    K1, K2, L = 0.01, 0.03, 255
    mu_x, mu_y = np.mean(original), np.mean(restored)
    sigma_x_sq, sigma_y_sq = np.var(original), np.var(restored)
    sigma_xy = np.cov(original.flatten(), restored.flatten())[0, 1]
    C1, C2 = (K1 * L) ** 2, (K2 * L) ** 2
    ssim_value = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x_sq + sigma_y_sq + C2))
    return ssim_value

# Fonction pour dessiner un masque
drawing = False
mask = None
brush_size = 15

def draw_mask(event, x, y, flags, param):
    global mask, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(mask, (x, y), brush_size, (0, 0, 0), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

# Chargement et filtrage de l'image
image = cv2.imread("Assets/25AvecBruit.jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Impossible de charger l'image.")
    exit(-1)

# Fichiers pour sauvegarder les métriques
ssim_file = open("ssim_values.dat", "w")
psnr_file = open("psnr_values.dat", "w")

# Fonction d'écriture des métriques
def save_metrics(step, img, ref_img):
    psnr = calculate_psnr(ref_img, img)
    ssim = calculate_ssim(ref_img, img)
    ssim_file.write(f"{step} {ssim}\n")
    psnr_file.write(f"{step} {psnr}\n")
    print(f"Etape {step} - PSNR: {psnr} dB, SSIM: {ssim}")

# Application de filtres
kernel_size, sigma, diameter = (7, 7), 0, 15
filtered_image = cv2.GaussianBlur(image, kernel_size, sigma)
save_metrics(1, filtered_image, image)

# Inpainting
height, width = filtered_image.shape[:2]
mask = np.ones((height, width), dtype=np.uint8) * 255

# Créer la fenêtre et utiliser setMouseCallback pour dessiner un masque
cv2.namedWindow("Draw Mask")
cv2.setMouseCallback("Draw Mask", draw_mask)

while True:
    img_with_mask = cv2.cvtColor(filtered_image, cv2.COLOR_GRAY2BGR)
    mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    combined_display = cv2.addWeighted(img_with_mask, 0.7, mask_display, 0.3, 0)
    
    cv2.imshow("Draw Mask", combined_display)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        cv2.imwrite('Assets/mask.jpg', mask)
        print("Masque sauvegardé sous 'Assets/mask.jpg'")
        
        # Appliquer l'inpainting
        mask_inverted = cv2.bitwise_not(mask)
        inpainted_img = cv2.inpaint(filtered_image, mask_inverted, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        save_metrics(2, inpainted_img, image)
        
        # Egalisation d'histogramme
        equalized_image = cv2.equalizeHist(inpainted_img)
        save_metrics(3, equalized_image, image)

        # Déconvolution Wiener (chaque 10 itérations comptent comme une seule étape)
        noise_var, signal_var = estimate_noise_and_signal_variance(image, inpainted_img)
        kernel = np.ones((3, 3)) / 25
        deconvolved_image = inpainted_img
        deconv_step = 4  # Numéro de l'étape de la déconvolution
        
        for i in range(0, 30, 10):  # Itérer par blocs de 10
            for _ in range(10):  # 10 itérations de déconvolution
                deconvolved_image = wiener_deconvolution(deconvolved_image, kernel, noise_var, signal_var)
            save_metrics(deconv_step, deconvolved_image, image)  # Sauvegarder les métriques pour chaque bloc de 10 itérations
            deconv_step += 1

        # Expansion dynamique
        expanded_image = dynamic_range_expansion(deconvolved_image)
        save_metrics(deconv_step, expanded_image, image)

        # Fermeture des fichiers de métriques
        ssim_file.close()
        psnr_file.close()
        
        break
    
    elif key == ord('q'):
        break

cv2.destroyAllWindows()
