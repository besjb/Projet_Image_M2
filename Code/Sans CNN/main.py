import cv2
import numpy as np

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


mask = None
drawing = False
brush_size = 5

def draw_mask(event, x, y, flags, param):
    global mask, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(mask, (x, y), brush_size, (0, 0, 0), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

image = cv2.imread("Assets/28.jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Impossible de charger l'image.")
    exit(-1)

damaged_img = image.copy()
height, width = damaged_img.shape[:2]
mask = np.ones((height, width), dtype=np.uint8) * 255 

cv2.namedWindow("Draw Mask")
cv2.setMouseCallback("Draw Mask", draw_mask)

while True:
    damaged_img_color = cv2.cvtColor(damaged_img, cv2.COLOR_GRAY2BGR)

    mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    combined_display = cv2.addWeighted(damaged_img_color, 0.5, mask_display, 0.5, 0)
    cv2.imshow("Draw Mask", combined_display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        cv2.imwrite('Assets/mask.jpg', mask)
        print("Mask saved as 'Assets/mask.jpg'")

        mask_inverted = cv2.bitwise_not(mask)
        inpainted_img = cv2.inpaint(damaged_img, mask_inverted, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        cv2.imshow("Inpainted Image", inpainted_img)
        cv2.imwrite('Assets/inpainted_image.jpg', inpainted_img)
        print("Inpainted image saved as 'Assets/inpainted_image.jpg'")

        # Expansion dynamique
        expanded_image = dynamic_range_expansion(inpainted_img)
        cv2.imshow("Expansion dynamique de l'image (apres deconvolution)", expanded_image)

        # Calcul des métriques de qualité entre image originale et l'image restaurée
        psnr_value = calculate_psnr(image, expanded_image)
        ssim_value = calculate_ssim(image, expanded_image)

        print(f"PSNR: {psnr_value} dB")
        print(f"SSIM: {ssim_value}")

    elif key == ord('q'):
        break

cv2.destroyAllWindows()
