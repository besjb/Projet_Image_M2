import cv2
import numpy as np

# Etape 1 : Pretraitement
def preprocess_image(image):
    # Convertir en niveaux de gris
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Appliquer un filtre de netete (soustraction de la moyenne locale)
    noyau = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    nettoye = cv2.filter2D(gris, -1, noyau)
    
    return nettoye

# Etape 2 : Creer et appliquer un ensemble de filtres passe-bande orientes
def filtres_passe_bande(image, nb_filtres=12):
    filtres = []
    pas_angle = 180 / nb_filtres
    for i in range(nb_filtres):
        angle = i * pas_angle
        # Creer le noyau de filtre oriente en utilisant une rotation
        noyau = cv2.getDerivKernels(1, 0, 1, 1, angle)
        noyau_dx = noyau[0]  # Noyau derive premiere (dx)
        noyau_dy = noyau[1]  # Noyau derive seconde (dy)

        noyau_combine = np.outer(noyau_dx, noyau_dy)
        filtres.append(noyau_combine)
        
    # Appliquer tous les filtres et combiaison des resultats
    images_filtrees = []
    for noyau in filtres:
        image_filtree = cv2.filter2D(image, -1, noyau)
        images_filtrees.append(image_filtree)
    
    return images_filtrees

# Etape 3 : Seuiler l'image pour creer un masque binaire
def seuil_image(image_filtree):
    _, masque_binaire = cv2.threshold(image_filtree, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return masque_binaire

# Etape 4 : Appliquer la transformation de Hough pour detecter les lignes (rayures)
def detecter_lignes(masque_binaire):
    lignes = cv2.HoughLinesP(masque_binaire, 1, np.pi / 180, threshold=150, minLineLength=10, maxLineGap=5)
    return lignes

# Etape 5 : Dessin des contours (detection des rayures)
def dessiner_contours(image, lignes):
    masque = np.zeros_like(image)
    if lignes is not None:
        for ligne in lignes:
            x1, y1, x2, y2 = ligne[0]
            cv2.line(masque, (x1, y1), (x2, y2), (255), 2)
    return masque

# Etape 6 : Remplir les rayures dans le masque (s'assurer qu'elles sont blanches)
def remplir_rayures(masque):

    taille_noyau = (5, 5)
    noyau = np.ones(taille_noyau, np.uint8)
    masque_rempli = cv2.morphologyEx(masque, cv2.MORPH_CLOSE, noyau)
    
    masque_dilate = cv2.dilate(masque_rempli, noyau, iterations=3)
    return masque_dilate

# Fonction principale pour combiner toutes les etapes
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


image = cv2.imread('Assets/Test7.png')
masque_rempli = restaurer_image(image)

cv2.imshow('Masque des Rayures', masque_rempli)
cv2.waitKey(0) 
cv2.destroyAllWindows() 

masque_rempli = np.uint8(masque_rempli > 0) * 255

cv2.imwrite('masque_rayures_restaure_bw.png', masque_rempli)

