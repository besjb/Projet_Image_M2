import cv2

chemin_image = 'Assets/Test.jpg'

image = cv2.imread(chemin_image)

if image is None:
    print("Erreur : impossible de charger l'image.")
else:
    cv2.imshow("Image", image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
