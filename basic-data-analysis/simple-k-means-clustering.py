import os
import cv2
import numpy as np
from sklearn.cluster import KMeans

# Configuration
input_folder = r"path-to-input"
output_folder = r"path-to-output"
n_clusters = 5 # Change to desired number of clusters

os.makedirs(output_folder, exist_ok=True)


def apply_kmeans_fixed_colors(image, n_clusters=2):
    pixel_values = image.reshape((-1, 3)).astype(np.float32)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(pixel_values)

    # Assign fixed grayscale colors
    grayscale_palette = np.array([[v, v, v] for v in np.linspace(0, 255, n_clusters).astype(np.uint8)])
    segmented_image = grayscale_palette[labels].reshape(image.shape)

    return segmented_image

# Process each image
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        img_path = os.path.join(input_folder, filename)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        clustered_img = apply_kmeans_fixed_colors(image, n_clusters)

        # Convert back to BGR for saving with OpenCV
        clustered_img_bgr = cv2.cvtColor(clustered_img, cv2.COLOR_RGB2BGR)
        out_path = os.path.join(output_folder, f'clustered_{filename}')
        cv2.imwrite(out_path, clustered_img_bgr)

        print(f"Saved clustered image to {out_path}")
