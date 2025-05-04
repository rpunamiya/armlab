import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def average_hsv(image_path):
    """ Extracts the average HSV values from an image. """
    image = cv2.imread(image_path)
    if image is None:
        return None  # Handle case where image cannot be loaded
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return np.mean(hsv_image, axis=(0, 1))  # Compute average (H, S, V)

def process_images(folder_path, max_images=100):
    """ Processes up to `max_images` images in a folder and extracts their average HSV values. """
    hsv_values = []
    image_names = []

    for filename in sorted(os.listdir(folder_path))[:max_images]:  # Limit to 100 images
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(folder_path, filename)
            avg_hsv = average_hsv(image_path)
            if avg_hsv is not None:
                hsv_values.append(avg_hsv)
                image_names.append(filename)

    return np.array(hsv_values), image_names

def plot_hsv_3d(hsv_values_1, hsv_values_2, image_names_1, image_names_2):
    """ Plots HSV values from two folders in a 3D scatter plot. """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Extract HSV components
    h1, s1, v1 = hsv_values_1[:, 0], hsv_values_1[:, 1], hsv_values_1[:, 2]
    h2, s2, v2 = hsv_values_2[:, 0], hsv_values_2[:, 1], hsv_values_2[:, 2]

    # Plot Folder 1 in red, Folder 2 in blue
    ax.scatter(h1, s1, v1, c='red', marker="o", label="Folder 1", edgecolor="black", s=50)
    ax.scatter(h2, s2, v2, c='blue', marker="o", label="Folder 2", edgecolor="black", s=50)

    ax.set_xlabel("Hue (H)")
    ax.set_ylabel("Saturation (S)")
    ax.set_zlabel("Value (V)")
    ax.set_title("3D Scatter Plot of HSV Values")
    ax.legend()

    plt.show()

# Set folder paths
folder_1 = r"C:\Users\rpuna\OneDrive - Stanford\Research\ARMLab\minc-2500\images\wood"   # Change to your actual path
folder_2 = r"C:\Users\rpuna\OneDrive - Stanford\Research\ARMLab\minc-2500\images\brick"  # Change to your actual path

# Extract HSV values
hsv_values_1, image_names_1 = process_images(folder_1, max_images=100)
hsv_values_2, image_names_2 = process_images(folder_2, max_images=100)

# Ensure both folders contain images before plotting
if len(hsv_values_1) > 0 and len(hsv_values_2) > 0:
    plot_hsv_3d(hsv_values_1, hsv_values_2, image_names_1, image_names_2)
else:
    print("Error: One or both folders contain fewer than 100 valid images.")
