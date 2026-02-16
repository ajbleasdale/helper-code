import cv2
import matplotlib.pyplot as plt


def plot_rgb_histogram(image_path, bins=256):
    # Read image (OpenCV loads as BGR)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Convert to RGB for correctness
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Split channels
    r, g, b = cv2.split(image_rgb)

    # Plot histograms
    plt.figure(figsize=(8, 5))
    plt.hist(r.ravel(), bins=bins, color='red', alpha=0.5, label='Red')
    plt.hist(g.ravel(), bins=bins, color='green', alpha=0.5, label='Green')
    plt.hist(b.ravel(), bins=bins, color='blue', alpha=0.5, label='Blue')

    plt.xlabel("Pixel intensity")
    plt.ylabel("Pixel count")
    plt.title("RGB Histogram")
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_rgb_histogram(r"path-to-image")
