import cv2
import numpy as np

image = cv2.imread("../Assets/27.jpg")
if image is None:
    print("Impossible de charger l'image.")
    exit(-1)

kernel_size = 15
filtered_image = cv2.medianBlur(image, kernel_size)

cv2.imshow("Image originale", image)
cv2.imshow("Image filtree (Median)", filtered_image)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cv2.destroyAllWindows()