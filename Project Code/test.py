import os
import cv2
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


def lossFunc(y_true, y_pred):
    w_mse = 0.4
    w_ssim = 0.6

    mse = tf.reduce_mean(tf.square(y_true - y_pred))

    ssim_value = tf.image.ssim(y_true, y_pred, max_val=1.0)
    ssim_loss = 1.0 - tf.reduce_mean(ssim_value)

    return (w_mse * mse) + (w_ssim * ssim_loss)

def get_score_per_image(originals, reconstructions):

    mse = np.mean(np.square(originals - reconstructions), axis=(1, 2, 3))
    ssim_scores = []
    for i in range(originals.shape[0]):
        ssim_score = ssim(originals[i], reconstructions[i], channel_axis=-1, data_range=1.0, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, win_size=11)
        ssim_scores.append(ssim_score)
    
    ssim_scores = np.array(ssim_scores)
    scores = (1 - ssim_scores) * 0.6 + mse * 0.4
    return (1 - ssim_scores) * 0.6 + mse * 0.4

def get_mse_per_image(originals, reconstructions):
    mse = np.mean(np.square(originals - reconstructions), axis=(1, 2, 3))
    return  mse

saved_fig_index = 0

class AnomalyDetector:
    def __init__(self, model_path, input_shape=(128, 128, 3)):
        """
        Args:
            model_path: Path to the .h5 file.
            threshold: The MSE value cutoff. Errors above this are Malignant.
            input_shape: The dimensions the model expects.
        """
        X_benign = np.load('TrainingDataBen.npy')
        X_malignant = np.load('TestDataMal.npy')
        self.model = tf.keras.models.load_model(model_path, compile=False, custom_objects={'lossFunc': lossFunc})
        
        reconstructions_benign = self.model.predict(X_benign)
        reconstructions_malignant = self.model.predict(X_malignant)
        score_benign = get_score_per_image(X_benign, reconstructions_benign)
        score_malignant = get_score_per_image(X_malignant, reconstructions_malignant)
       
        mean_loss = np.mean(score_benign)
        print("Training Set - Benign Mean Score: ", mean_loss)
        print("Training Set - Malignant Mean Score: ", np.mean(score_malignant))

        std_loss = np.std(score_benign)
        print("Training Set - Benign Std Dev: ", std_loss)
        std_loss_malignant = np.std(score_malignant)
        print("Training Set - Malignant Std Dev: ", std_loss_malignant)
        threshold_value = mean_loss + (1.2 * std_loss)
        self.threshold = threshold_value
        self.input_shape = input_shape
        print(f"Model loaded from {model_path}")
        print(f"Anomaly Threshold set to: {self.threshold}")

        score_malignant_above_threshold = np.sum(score_malignant > self.threshold)
        print(f"Malignant images above threshold: {score_malignant_above_threshold}/{len(score_malignant)} = {score_malignant_above_threshold/len(score_malignant):.2%}")


        score_benign_below_threshold = np.sum(score_benign <= self.threshold)
        print(f"Benign images below threshold: {score_benign_below_threshold}/{len(score_benign)} = {score_benign_below_threshold/len(score_benign):.2%}")


    def hair_removal(self,image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # > 17 -> thicker hairs.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))

        blackHat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

        # 10 -> 20 if erases moles and makes dataset bad.
        _, mask = cv2.threshold(blackHat, 10, 255, cv2.THRESH_BINARY)

        # median like repaint mask
        hair_removed = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

        return hair_removed
    def preprocess(self, img_path):
        """Loads and prepares an image exactly how the model expects it."""

        img_array = cv2.imread(img_path)
        if img_array is None: return None
        hair_removed_array = self.hair_removal(img_array)
        new_array = cv2.cvtColor(hair_removed_array, cv2.COLOR_BGR2RGB)
        new_array = cv2.resize(new_array, (self.input_shape[1], self.input_shape[0]))
        new_array = new_array.astype('float32') / 255.0
        new_array = np.expand_dims(new_array, axis=0)
        return new_array, hair_removed_array

    def classify(self, img_path, original_status=None, plot=True):
        img_input, original_img = self.preprocess(img_path)
        reconstruction = self.model.predict(img_input, verbose=0)


        ssim_score, diff_map = ssim(img_input[0], reconstruction[0], 
                           channel_axis=-1, 
                           full=True, 
                           data_range=1.0, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, win_size=11)
        
        mse = np.mean(np.square(img_input - reconstruction))

        score = (1 - ssim_score) * 0.6 + mse * 0.4
        is_malignant = score > self.threshold

        diff = np.abs(img_input[0] - reconstruction[0])
        diff_gray = np.mean(diff, axis=-1)
        result = {
            "original_status": original_status,
            "status": "MALIGNANT" if is_malignant else "BENIGN",
            "score": score,
            "diff_map": diff_gray,
            "threshold": self.threshold,
            "is_anomaly": is_malignant
        }

        if plot:
            self._visualize(original_img, reconstruction[0], result)

        return result

    def _visualize(self, original, reconstructed, result):
        """Helper to plot the input vs output and the error map."""        
        
        color = 'red' if result['is_anomaly'] else 'green'

        plt.figure(figsize=(10, 3))

        plt.subplot(1, 3, 1)
        plt.imshow(original)
        plt.title("New Input Image")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(reconstructed)
        plt.title("Model Reconstruction")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(result['diff_map'], cmap='jet')
        plt.title(f"{result['status']}\nScore: {result['score']:.4f}, Original: {result['original_status']}", color=color, fontweight='bold')
        plt.axis('off')
        plt.colorbar()

        plt.tight_layout()

        global saved_fig_index
        mke_dir = "test_figs"
        if not os.path.exists(mke_dir):
            os.makedirs(mke_dir)
        plt.savefig(f"test_figs/output_{saved_fig_index}.png")
        saved_fig_index += 1



model_path = 'SkinCancerAutoencoder.h5'
input_shape = (128, 128, 3)
detector = AnomalyDetector(model_path, input_shape=input_shape)



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

ALL_CLASSES = BENIGN_CLASSES + MALIGNANT_CLASSES

from pathlib import Path

image_dir = Path(input("Enter Dataset path: ").strip())

if not image_dir.is_dir():
    raise ValueError("Invalid folder path")

print("Selected folder:", image_dir)
image_dir = os.path.join(image_dir.resolve(), "Test")


benign_count = 0
total_benign = 0
malignant_count = 0
total_malignant = 0

for class_name in BENIGN_CLASSES:
    class_path = os.path.join(image_dir, class_name)
    for img_file in os.listdir(class_path):
        img_path = os.path.join(class_path, img_file)
        result = detector.classify(img_path, original_status="BENIGN", plot=False)
        total_benign += 1
        if not result['is_anomaly']:
            benign_count += 1
for class_name in MALIGNANT_CLASSES:
    class_path = os.path.join(image_dir, class_name)
    for img_file in os.listdir(class_path):
        img_path = os.path.join(class_path, img_file)
        result = detector.classify(img_path, original_status="MALIGNANT", plot=False)
        total_malignant += 1
        if result['is_anomaly']:
            malignant_count += 1

print(f"Test Set - Malignant Accuracy: {malignant_count}/{total_malignant} = {malignant_count/total_malignant:.2%}")
print(f"Test Set - Benign Accuracy: {benign_count}/{total_benign} = {benign_count/total_benign:.2%}")

for class_name in ALL_CLASSES:
    class_path = os.path.join(image_dir, class_name)
    for img_file in os.listdir(class_path):
        img_path = os.path.join(class_path, img_file)
        original_status = "MALIGNANT" if class_name in MALIGNANT_CLASSES else "BENIGN"
        result = detector.classify(img_path, original_status=original_status, plot=True)