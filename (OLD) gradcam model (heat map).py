# activate venv first in terminal:
# cd C:\Users\mchickering27\Downloads\datscan_pd_classifier
# venv\Scripts\activate
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model
from tensorflow.keras import models
from tensorflow.keras.layers import Input

#Image settings
IMG_SIZE = 128
MODEL_PATH = "dat_scan_model.h5"


#Loading the trained model
model = load_model(MODEL_PATH)

#Wrap in new functional model to define input
new_input = Input(shape=(128, 128, 1))
new_outputs = model(new_input)
model = tf.keras.Model(inputs=new_input, outputs=new_outputs)

#File picker to choose image
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(title="Select a DaTSCAN Image")

if not file_path:
    print("No file selected.")
    exit()

#Load and reprocess the image picked
img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img_input = img.reshape(1, IMG_SIZE, IMG_SIZE, 1) / 255.0

#Make prediction
preds = model.predict(img_input)[0]
pred_label = np.argmax(preds)
confidence = preds[pred_label]

print(f"\nPrediction: {'PD' if pred_label == 1 else 'Non-PD'} ({confidence:.2%} confidence)")

#Gradcam logic
last_conv_layer = model.get_layer(index=0)

heatmap_model = models.Model([model.input], [last_conv_layer.output, model.output])

#Gradcam computation
with tf.GradientTape() as tape:
    img_tensor = tf.convert_to_tensor(img_input, dtype=tf.float32)
    tape.watch(img_tensor)
    conv_output, predictions = heatmap_model(img_tensor)
    class_channel = predictions[:, pred_label]

#Compute gradients
grads = tape.gradient(class_channel, conv_output)
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # ✅ no more None


conv_output = conv_output[0]
heatmap = conv_output @ pooled_grads[..., tf.newaxis]
heatmap = tf.squeeze(heatmap)

#normalize heatmap
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)

#Overlay heatmap on original image
heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
heatmap = np.uint8(255 * heatmap)
heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
overlay = cv2.addWeighted(img_bgr, 0.5, heatmap_color, 0.5, 0)

cv2.imwrite("gradcam_visualization.png", overlay)
print("Saved Grad-CAM image to gradcam_visualization.png")


#how Grad-CAM result in a window
cv2.imshow("Grad-CAM", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()

