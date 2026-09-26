import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Function to display and save images
def show_comparison(original, sobel, prewitt, canny):

    plt.figure(figsize=(12, 10))

    # Original image
    plt.subplot(2, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original grayscale image')
    plt.axis('off')

    # Sobel edge detection
    plt.subplot(2, 2, 2)
    plt.imshow(sobel, cmap='gray')
    plt.title('Sobel edge detection')
    plt.axis('off')

    # Prewitt edge detection
    plt.subplot(2, 2, 3)
    plt.imshow(prewitt, cmap='gray')
    plt.title('Prewitt edge detection')
    plt.axis('off')

    # Canny edge detection
    plt.subplot(2, 2, 4)
    plt.imshow(canny, cmap='gray')
    plt.title('Canny edge detection')
    plt.axis('off')

    plt.tight_layout()

    # Create output folder
    os.makedirs("output", exist_ok=True)

    # Save output image
    plt.savefig(
        "output/edge_detection_comparison.png",
        bbox_inches='tight',
        dpi=300
    )

    plt.show()


# Read an image
image = cv2.imread('nature.JPG', 0)

if image is None:
    print('Error: image not found')
else:

    # Sobel operator
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Combine Sobel gradients
    sobel_combined = cv2.magnitude(sobelx, sobely)

    # Prewitt operator
    kernelx = np.array([
        [1, 1, 1],
        [0, 0, 0],
        [-1, -1, -1]
    ])

    kernely = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ])

    # Apply Prewitt kernels
    prewittx = cv2.filter2D(image, -1, kernelx)
    prewitty = cv2.filter2D(image, -1, kernely)

    # Combine Prewitt gradients
    prewitt_combined = cv2.addWeighted(
        prewittx, 0.5,
        prewitty, 0.5,
        0
    )

    # Canny edge detection
    canny = cv2.Canny(image, 100, 200)

    # Display and save comparison
    show_comparison(
        image,
        np.uint8(np.absolute(sobel_combined)),
        prewitt_combined,
        canny
    )

    print("Output saved as: output/edge_detection_comparison.png")