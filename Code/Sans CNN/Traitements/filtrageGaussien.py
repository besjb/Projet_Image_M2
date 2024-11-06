import cv2
import numpy as np

image = cv2.imread("Assets/Test.jpg")
if image is None:
    print("Impossible de charger l'image.")
    exit(-1)

kernel_size = (15, 15)
sigma = 0
filtered_image = cv2.GaussianBlur(image, kernel_size, sigma)

cv2.imshow("Image originale", image)
cv2.imshow("Image filtrée (Gaussien)", filtered_image)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cv2.destroyAllWindows()
