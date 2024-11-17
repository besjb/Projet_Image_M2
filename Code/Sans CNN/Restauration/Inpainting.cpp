import cv2
import numpy as np

mask = None
drawing = False
brush_size = 25

def draw_mask(event, x, y, flags, param):
    global mask, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(mask, (x, y), brush_size, (0, 0, 0), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False 

damaged_img = cv2.imread("Assets/Test10.jpg")

if damaged_img is None:
    print("Error: Image not found. Please check the path.")
    exit()

height, width = damaged_img.shape[:2]
mask = np.ones((height, width), dtype=np.uint8) * 255

cv2.namedWindow("Dessiner le masque")
cv2.setMouseCallback("Dessiner le masque", draw_mask)

while True:
    mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    combined_display = cv2.addWeighted(damaged_img, 0.5, mask_display, 0.5, 0) 
    cv2.imshow("Dessiner le masque", combined_display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        cv2.imwrite('Assets/mask.jpg', mask)
        print("Masque sauvegardé as 'Assets/mask.jpg'")

        mask_inverted = cv2.bitwise_not(mask)
        kernel = np.ones( (7,7) , np.uint8)
        mask = cv2.dilate(src=thresh1, kernel=kernel, iterations=1)
        inpainted_img = cv2.inpaint(damaged_img, mask_inverted, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        cv2.imshow("Inpainted Image", inpainted_img)
        cv2.imwrite('Assets/inpainted_image.jpg', inpainted_img)
        print("Inpainted image saved as 'Assets/inpainted_image.jpg'")

    elif key == ord('q'):
        break

cv2.destroyAllWindows()
