import cv2
import numpy as np
from scipy.ndimage import convolve

# Fonction pour calculer la magnitude du gradient avec Sobel
def gradient_magnitude(image):
    # Application du filtre Sobel pour calculer les gradients
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    grad_x = convolve(image, sobel_x)
    grad_y = convolve(image, sobel_y)

    # Magnitude du gradient
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return gradient_magnitude

# Fonction pour détecter et masquer les visages dans l'image
def detect_faces(image):
    # Utiliser le chemin absolu vers le fichier Haar Cascade XML
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # Remplacez par le chemin complet
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(40, 40))
    
    # Masquer les visages dans l'image
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), -1)  # Noir pour masquer les visages

# Charger l'image
image_path = 'Assets/Test7.png'  # Vérifiez que le chemin vers l'image est correct
image = cv2.imread(image_path)

# Vérification si l'image a bien été chargée
if image is None:
    print(f"Erreur lors du chargement de l'image à partir de {image_path}. Vérifiez le chemin.")
else:
    image_resized = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
    detect_faces(image_resized)
    gray_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    gSize = 27
    gaussian_filtered = cv2.GaussianBlur(gray_image, (gSize, gSize), gSize / 2)

    gradient_magnitude_image = gradient_magnitude(gaussian_filtered)
    threshold_value = 0.3 * np.max(gradient_magnitude_image)
    binary_image = gradient_magnitude_image > threshold_value

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # Remplacer MORPH_DISK par MORPH_ELLIPSE
    closed_image = cv2.morphologyEx(binary_image.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    # Appliquer l'analyse de connexité des composants (CCL - Connexité des composants)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed_image)
    filtered_image = np.zeros_like(closed_image)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 500:
            filtered_image[labels == i] = 255

    closed_image_2 = cv2.morphologyEx(filtered_image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
    filled_image = cv2.morphologyEx(closed_image_2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    kernel_opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened_image = cv2.morphologyEx(filled_image, cv2.MORPH_OPEN, kernel_opening)

    filled_image2 = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    kernel_opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened_image_after_close = cv2.morphologyEx(filled_image2, cv2.MORPH_OPEN, kernel_opening)
    filled_image3 = cv2.morphologyEx(opened_image_after_close, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    cv2.imshow('Original Image', image)
    cv2.imshow('Detected Scratch Mask', filled_image3)

    cv2.imwrite('detected_scratch_mask.png', filled_image3)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

