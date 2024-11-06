import cv2
import numpy as np

image = cv2.imread("Assets/Test7.png", cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Impossible de charger l'image.")
    exit(-1)

cv2.imshow("Image originale", image)

equalized_image = cv2.equalizeHist(image)

cv2.imshow("Image égalisée", equalized_image)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cv2.destroyAllWindows()
