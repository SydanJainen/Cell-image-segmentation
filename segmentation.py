import os
import cv2
import uuid
import albumentations as A
from matplotlib import pyplot as plt
from glob import glob

DB_PATH = 'database/'
TRAIN_PATH = os.path.join(DB_PATH, 'train/')
LABEL_PATH = os.path.join(DB_PATH, 'label/')

DESTINATION_PATH = 'database/augmented/'
DESTINATION_LABEL_PATH = os.path.join(DESTINATION_PATH, 'label/')
DESTINATION_TRAIN_PATH = os.path.join(DESTINATION_PATH, 'train/')

# Ensure destination directories exist
os.makedirs(DESTINATION_TRAIN_PATH, exist_ok=True)
os.makedirs(DESTINATION_LABEL_PATH, exist_ok=True)

# Define transformations
transform = A.Compose([
    A.RandomRotate90(),
    A.Flip(),
    A.Transpose(),
    A.CLAHE(clip_limit=2),
    A.OneOf([
        A.MotionBlur(p=0.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.Blur(blur_limit=3, p=0.1),
    ], p=0.2),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.2),
    A.HueSaturationValue(p=0.3),
])

# Get all image and mask paths
image_paths = glob(os.path.join(TRAIN_PATH, '*.png'))
mask_paths = glob(os.path.join(LABEL_PATH, '*.png'))

# Function to apply augmentations and save results
def augment_and_save(image_path, mask_path):
    assert os.path.exists(image_path), f"Image path does not exist: {image_path}"
    assert os.path.exists(mask_path), f"Mask path does not exist: {mask_path}"
    #mask and image should have the same name
    assert os.path.basename(image_path) == os.path.basename(mask_path), "Image and mask filenames do not match."    
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for i in range(10):  # Apply 4 different augmentations
        # rezie image and mask
        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256))
        augmented = transform(image=image, mask=mask)
        augmented_image = augmented['image']
        augmented_mask = augmented['mask']

        # Convert images back to BGR before saving
        augmented_image_bgr = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
        augmented_mask_bgr = cv2.cvtColor(augmented_mask, cv2.COLOR_RGB2BGR)

        # Construct save paths
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        save_image_path = os.path.join(DESTINATION_TRAIN_PATH, f"{base_filename}_aug_{i}.png")
        save_mask_path = os.path.join(DESTINATION_LABEL_PATH, f"{base_filename}_aug_{i}.png")

        # Save augmented images and masks
        cv2.imwrite(save_image_path, augmented_image_bgr)
        cv2.imwrite(save_mask_path, augmented_mask_bgr)

# Process each image and its corresponding mask
for image_path in image_paths:
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    mask_path = os.path.join(LABEL_PATH, f"{base_filename}.png")
    if os.path.exists(mask_path):
        augment_and_save(image_path, mask_path)

print("Augmentation and saving completed.")
