import cv2
import numpy as np

image = cv2.imread("../Assets/27.jpg")
if image is None:
    print("Impossible de charger l'image.")
    exit(-1)

kernel_size = (13, 13)
sigma = 0
filtered_image = cv2.GaussianBlur(image, kernel_size, sigma)

cv2.imshow("Image originale", image)
cv2.imshow("Image filtree (Gaussien)", filtered_image)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cv2.destroyAllWindows()
