import os
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = r'path/to/qt/plugins/platforms'

import albumentations as A
import matplotlib.pyplot as plt

import cv2

# Define transformations
transform = A.Compose([
    A.Rotate(limit=[-90, 90], p=1.0)
])

# Load image
image = cv2.imread('/home/maciek/Documents/Studia/Magisterka/master-thesis/coco_dataset/images_new/4.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Apply transformations
augmented_images = [transform(image=image)['image'] for _ in range(2)]  # Augment 5 times

# Plot original and augmented images
plt.figure(figsize=(12, 6))
plt.subplot(2, 3, 1)
plt.imshow(image)
plt.title('Original')

for i in range(2):
    plt.subplot(2, 3, i + 2)
    plt.imshow(augmented_images[i])
    plt.title(f'Rotate {i+1}')

plt.tight_layout()
plt.show()