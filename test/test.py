import cv2
import numpy as np

# Load the original image and segmentation mask
image = cv2.imread('/unet-semantic/test/1gVcuyz-6to_1.png')
mask = cv2.imread('/unet-semantic/test/helicopter1gVcuyz-6to_output.jpg')

# Convert the mask to a colored overlay
overlay = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

# Adjust the transparency of the overlay
alpha = 0.5  # Set the desired transparency value (0.0 - fully transparent, 1.0 - fully opaque)
overlay = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

# Display or save the final result
cv2.imshow('Segmentation Result', overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()