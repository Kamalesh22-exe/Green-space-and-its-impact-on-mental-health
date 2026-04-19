import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image = cv2.imread("campus_area.jpg")

# Convert BGR to RGB (for display)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define green color range (tunable)
lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])

# Create mask
mask = cv2.inRange(hsv, lower_green, upper_green)

# Count pixels
green_pixels = np.count_nonzero(mask)
total_pixels = mask.size

green_percentage = (green_pixels / total_pixels) * 100

print(f"Green Cover Percentage: {green_percentage:.2f}%")

# Show original and mask
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(image_rgb)
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Detected Green Areas")
plt.imshow(mask, cmap="gray")
plt.axis("off")

plt.show()