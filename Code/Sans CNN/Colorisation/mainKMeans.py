import cv2
import numpy as np

image = cv2.imread("Assets/Test.jpg")
if image is None:
    print("Impossible de charger l'image.")
    exit(-1)

K = int(input("Choisir un nombre K : "))
if K < 2:
    print("K doit être supérieur ou égal à 2.")
    exit(-1)

data = image.reshape((-1, 3)).astype(np.float32)

# K-means
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
_, labels, centers = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Moyennes des couleurs / cluster
cluster_colors = np.zeros_like(centers, dtype=np.float32)
cluster_counts = np.zeros(K, dtype=int)

# Accumuler les couleurs de chaque pixel / cluster
for i in range(len(labels)):
    cluster_idx = labels[i][0]
    cluster_colors[cluster_idx] += data[i]
    cluster_counts[cluster_idx] += 1

# Calcul de la moyenne des couleurs / cluster
for i in range(K):
    if cluster_counts[i] > 0:
        cluster_colors[i] /= cluster_counts[i]

# Convertir les moyennes en valeurs entières
cluster_colors = np.uint8(cluster_colors)

# Appliquer les couleurs moyennes aux clusters dans l'image
result_image = cluster_colors[labels.flatten()]
color_image = result_image.reshape(image.shape)

cv2.imshow("Image originale", image)
cv2.imshow("Image colorisee avec moyennes des clusters", color_image)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cv2.destroyAllWindows()
