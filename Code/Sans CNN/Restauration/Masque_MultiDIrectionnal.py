import cv2
import numpy as np

# Fonction pour calculer le PSNR
def calculate_psnr(original, restored):
    return cv2.PSNR(original, restored)

# Fonction pour calculer le SSIM
def calculate_ssim(original, restored):
    K1, K2, L = 0.01, 0.03, 255
    mu_x, mu_y = np.mean(original), np.mean(restored)
    sigma_x_sq, sigma_y_sq = np.var(original), np.var(restored)
    sigma_xy = np.cov(original.flatten(), restored.flatten())[0, 1]
    C1, C2 = (K1 * L) ** 2, (K2 * L) ** 2
    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x_sq + sigma_y_sq + C2)
    return numerator / denominator

def preprocess_image(image):
    if len(image.shape) == 3:
        gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gris = image
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

image = cv2.imread("Assets/25.jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Impossible de charger l'image.")
    exit(-1)

masque_rempli = restaurer_image(image)

masque_rempli = np.uint8(masque_rempli > 0) * 255
if len(masque_rempli.shape) == 3:
    masque_rempli = cv2.cvtColor(masque_rempli, cv2.COLOR_BGR2GRAY)

inpainted_img = cv2.inpaint(image, masque_rempli, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

# Affichage des résultats
cv2.imshow("Inpainted Image", inpainted_img)
cv2.imshow("Masque", masque_rempli)

cv2.imwrite('Assets/inpainted_image.jpg', inpainted_img)

# Calcul du PSNR et du SSIM
psnr_value = calculate_psnr(image, inpainted_img)
ssim_value = calculate_ssim(image, inpainted_img)

print(f"PSNR de l'image restaurée : {psnr_value} dB")
print(f"SSIM de l'image restaurée : {ssim_value}")

cv2.waitKey(0)
cv2.destroyAllWindows()
