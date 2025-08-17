import os
import cv2
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# --- DATA LOADING ---
def load_images_and_masks(img_folder, mask_folder, img_size=(128,128)):
    images = []
    masks = []
    img_filenames = os.listdir(img_folder)
    for filename in img_filenames:
        img_path = os.path.join(img_folder, filename)
        mask_path = os.path.join(mask_folder, filename)  # filenames must match
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue
        img = cv2.resize(img, img_size)
        mask = cv2.resize(mask, img_size)
        img = img.astype(np.float32) / 255.
        mask = mask.astype(np.float32) / 255.
        img = np.expand_dims(img, axis=-1)
        mask = np.expand_dims(mask, axis=-1)
        images.append(img)
        masks.append(mask)
    X = np.array(images)
    Y = np.array(masks)
    return X, Y

# --- DEFINE UNET ---
def unet_model(input_size=(128,128,1)):
    inputs = Input(input_size)
    c1 = Conv2D(16, (3,3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(16, (3,3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2,2))(c1)

    c2 = Conv2D(32, (3,3), activation='relu', padding='same')(p1)
    c2 = Conv2D(32, (3,3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2,2))(c2)

    c3 = Conv2D(64, (3,3), activation='relu', padding='same')(p2)
    c3 = Conv2D(64, (3,3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2,2))(c3)

    c4 = Conv2D(128, (3,3), activation='relu', padding='same')(p3)
    c4 = Conv2D(128, (3,3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2,2))(c4)

    c5 = Conv2D(256, (3,3), activation='relu', padding='same')(p4)
    c5 = Conv2D(256, (3,3), activation='relu', padding='same')(c5)

    u6 = Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3,3), activation='relu', padding='same')(u6)
    c6 = Conv2D(128, (3,3), activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3,3), activation='relu', padding='same')(u7)
    c7 = Conv2D(64, (3,3), activation='relu', padding='same')(c7)

    u8 = Conv2DTranspose(32, (2,2), strides=(2,2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3,3), activation='relu', padding='same')(u8)
    c8 = Conv2D(32, (3,3), activation='relu', padding='same')(c8)

    u9 = Conv2DTranspose(16, (2,2), strides=(2,2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(16, (3,3), activation='relu', padding='same')(u9)
    c9 = Conv2D(16, (3,3), activation='relu', padding='same')(c9)
    outputs = Conv2D(1, (1,1), activation='sigmoid')(c9)
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- TRAINING ---
if __name__ == "__main__":
    img_folder = 'data/images'
    mask_folder = 'data/masks'
    X, Y = load_images_and_masks(img_folder, mask_folder)
    print(f"Loaded {X.shape[0]} images.")
    model = unet_model()
    model.summary()
    model.fit(X, Y, batch_size=2, epochs=10, validation_split=0.1)
    os.makedirs('models', exist_ok=True)
    model.save('models/pretrained_unet.h5')
    print("Model saved as 'models/pretrained_unet.h5'")
