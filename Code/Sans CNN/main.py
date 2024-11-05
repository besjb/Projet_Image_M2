import cv2
import numpy as np

mask = None
drawing = False
taille_pinceau = 5

def draw_mask(event, x, y, flags, param):
    global mask, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(mask, (x, y), taille_pinceau, (0, 0, 0), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False 

damaged_img = cv2.imread("Assets/Test10.jpg")

if damaged_img is None:
    print("Image non trouvé.")
    exit()

height, width = damaged_img.shape[:2]
mask = np.ones((height, width), dtype=np.uint8) * 255

cv2.namedWindow("Dessiner_Mask")
cv2.setMouseCallback("Dessiner_Mask", draw_mask)

while True:
    mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    combined_display = cv2.addWeighted(damaged_img, 0.5, mask_display, 0.5, 0) 
    cv2.imshow("Dessiner_Mask", combined_display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        cv2.imwrite('Assets/mask.jpg', mask)
        print("Mask save  'Assets/mask.jpg'")

        mask_inverted = cv2.bitwise_not(mask)

        inpainted_img = cv2.inpaint(damaged_img, mask_inverted, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        cv2.imshow("Image Inpainted", inpainted_img)
        cv2.imwrite('Assets/inpainted_image.jpg', inpainted_img)
        print("image Inpainted save 'Assets/inpainted_image.jpg'")

    elif key == ord('q'):  # Quit
        break

cv2.destroyAllWindows()
