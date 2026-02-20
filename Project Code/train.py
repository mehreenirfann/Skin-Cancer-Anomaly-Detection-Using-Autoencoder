import os
import cv2
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from pathlib import Path


IMG_SIZE = 128
BENIGN_CLASSES = [
    "nevus",
    "dermatofibroma",
    "pigmented benign keratosis",
    "seborrheic keratosis",
    "vascular lesion"
]

MALIGNANT_CLASSES = [
    "melanoma",
    "basal cell carcinoma",
    "squamous cell carcinoma",
    "actinic keratosis"
]

def hair_removal(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # > 17 -> thicker hairs.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))

    blackHat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # 10 -> 20 if erases moles and makes dataset bad.
    _, mask = cv2.threshold(blackHat, 10, 255, cv2.THRESH_BINARY)

    # median like repaint mask
    hair_removed = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

    return hair_removed

TrainingDataBen = []
TestDataMal = []
source_folders = ['Train']


image_dir = Path(input("Enter Dataset path: ").strip())

if not image_dir.is_dir():
    raise ValueError("Invalid folder path")


for source in source_folders:
    source_path = os.path.join(image_dir, source)

    for class_folder in os.listdir(source_path):
        class_folder_lower = class_folder.lower()
        class_path = os.path.join(source_path, class_folder)

        if not os.path.isdir(class_path):
            continue

        if class_folder_lower in BENIGN_CLASSES:
            target_list = TrainingDataBen
            label_type = "Benign"
        elif class_folder_lower in MALIGNANT_CLASSES:
            target_list = TestDataMal
            label_type = "Malignant"
        else:
            continue

        for img_name in tqdm(os.listdir(class_path), desc=f"{source}/{class_folder}"):
            try:
                img_path = os.path.join(class_path, img_name)
                img_array = cv2.imread(img_path)

                if img_array is None: continue

                hair_removed_array = hair_removal(img_array)
                img_array = cv2.cvtColor(hair_removed_array, cv2.COLOR_BGR2RGB)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

                target_list.append(new_array)
            except Exception:
                pass
X_benign = np.array(TrainingDataBen).astype('float32') / 255.0
X_malignant = np.array(TestDataMal).astype('float32') / 255.0

np.save('TrainingDataBen.npy', X_benign)
np.save('TestDataMal.npy', X_malignant)

def lossFunc(y_true, y_pred):
    w_mse = 0.4
    w_ssim = 0.6

    mse = tf.reduce_mean(tf.square(y_true - y_pred))

    ssim_value = tf.image.ssim(y_true, y_pred, max_val=1.0)
    ssim_loss = 1.0 - tf.reduce_mean(ssim_value)

    return (w_mse * mse) + (w_ssim * ssim_loss)


def Autoencoder(input_shape=(128, 128, 3)):

    input_img = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)

    shape_before_flattening = K.int_shape(x)

    x = layers.Flatten()(x)
    encoded = layers.Dense(128, activation='relu', name='bottleneck',activity_regularizer=tf.keras.regularizers.l1(10e-8))(x)

##decoder part
    x = layers.Dense(np.prod(shape_before_flattening[1:]), activation='relu')(encoded)
    x = layers.Reshape(shape_before_flattening[1:])(x)

    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x) # 128x128

    decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    return models.Model(input_img, decoded)

model = Autoencoder(input_shape=(IMG_SIZE, IMG_SIZE, 3))
model.summary()

model.compile(optimizer='adam', loss=lossFunc)

history = model.fit(
    X_benign, X_benign,
    epochs=100,
    batch_size=16,
    shuffle=True,
    validation_split=0.1,
    verbose=1
)

model.save("SkinCancerAutoencoder.h5")

reconstructions_benign = model.predict(X_benign)
reconstructions_malignant = model.predict(X_malignant)

def get_mse_per_image(originals, reconstructions):
    return np.mean(np.square(originals - reconstructions), axis=(1, 2, 3))

mse_benign = get_mse_per_image(X_benign, reconstructions_benign)
mse_malignant = get_mse_per_image(X_malignant, reconstructions_malignant)

print(f"Mean Benign Error: {np.mean(mse_benign)}")
print(f"Mean Malignant Error: {np.mean(mse_malignant)}")