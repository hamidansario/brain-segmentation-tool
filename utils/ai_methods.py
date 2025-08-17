from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

MODEL_PATH = 'models/pretrained_unet.h5'

# Load your trained U-Net once
model = load_model(MODEL_PATH)

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=(0, -1))
    return img

def segment_brain(img_path):
    input_img = preprocess_image(img_path)
    pred_mask = model.predict(input_img)[0, :, :, 0]
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
    mask_path = img_path.rsplit('.', 1)[0] + '_unet_mask.png'
    cv2.imwrite(mask_path, pred_mask)
    return mask_path

import cv2
import numpy as np

def overlay_mask_on_image(image_path, mask_path, alpha=0.5, output_path=None):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found: {image_path}")
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Mask not found: {mask_path}")
    # Resize mask to match image
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    color_mask = np.zeros_like(image)
    color_mask[:, :, 2] = binary_mask
    overlayed = cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)
    if output_path:
        cv2.imwrite(output_path, overlayed)
    return overlayed
