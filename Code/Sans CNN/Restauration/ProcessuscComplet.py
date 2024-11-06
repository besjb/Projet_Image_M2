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

mask = None
drawing = False
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

image = cv2.imread("Assets/Test10.jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Impossible de charger l'image.")
    exit(-1)

cv2.imshow("Image originale", image)
print("1: Flou Gaussien")
print("2: Flou Médian")
print("3: Flou Bilatéral")
print("q: Quitter")

key = cv2.waitKey(0) & 0xFF

if key == ord('1'):  # Flou Gaussien
    kernel_size = (7, 7)
    sigma = 0
    filtered_image = cv2.GaussianBlur(image, kernel_size, sigma)
    cv2.imshow("Image filtree (Gaussien)", filtered_image)

elif key == ord('2'):  # Flou Médian
    kernel_size = 7
    filtered_image = cv2.medianBlur(image, kernel_size)
    cv2.imshow("Image filtree (Median)", filtered_image)

elif key == ord('3'):  # Flou Bilatéral
    diameter = 15  
    sigma_color = 75  
    sigma_space = 75  
    filtered_image = cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)
    cv2.imshow("Image filtree (Bilateral)", filtered_image)

damaged_img = filtered_image.copy()
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

        # Égalisation d'histogramme
        equalized_image = cv2.equalizeHist(inpainted_img)
        cv2.imshow("Image egalisee", equalized_image)

        # Déconvolution
        noise_var, signal_var = estimate_noise_and_signal_variance(image, inpainted_img, np.ones((5, 5)) / 5)
        kernel = np.ones((3, 3)) / 5
        deconvolved_image = wiener_deconvolution(inpainted_img, kernel, noise_var, signal_var)
        deconvolved_image2 = wiener_deconvolution(deconvolved_image, kernel, noise_var, signal_var)

        cv2.imshow("Image Deconvoluee", deconvolved_image.astype(np.uint8))

        # Expansion dynamique
        expanded_image = dynamic_range_expansion(deconvolved_image2)
        cv2.imshow("Expansion dynamique de l'image (apres deconvolution)", expanded_image)

    elif key == ord('q'):  # Quitter
        break

cv2.destroyAllWindows()