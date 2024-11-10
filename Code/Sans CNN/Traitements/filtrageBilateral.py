import cv2
import numpy as np

image = cv2.imread("../Assets/27.jpg")
if image is None:
    print("Impossible de charger l'image.")
    exit(-1)

diameter = 15
sigma_color = 75  
sigma_space = 75  
filtered_image = cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)

cv2.imshow("Image originale", image)
cv2.imshow("Image filtree (Bilateral)", filtered_image)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cv2.destroyAllWindows()
