#keep tensorflow or switch to pytorch?? create own lib??
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
import os  #file path operations
import cv2  #iimage loading and processing
import numpy as np  #numerical operations and array handling
from sklearn.model_selection import train_test_split  #splitting data into training and validation sets
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt  #plotting (training curves later)

#configuration 
#set image size to resize all input scans to a uniform shape
IMG_SIZE = 128  # Each image will be resized to 128x128 pixels

#define base data directory where PD and Non PD folders are
DATA_DIR = "data"

CATEGORIES = ["Non_PD", "PD"]  # Class 0 = Non-PD, Class 1 = PD

#load data
def load_data():
    data = []  # List to hold all image/label pairs

    #loop through each category
    for label, category in enumerate(CATEGORIES):
        folder = os.path.join(DATA_DIR, category)  #full path to category folder
        for img_file in os.listdir(folder):  #loop through each image file in the folder
            try:
                img_path = os.path.join(folder, img_file)  #path to image file
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  #read image in grayscale
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  #resize image to standard size
                data.append((img, label))  #add image and label to  dataset
            except:
                print(f"Error loading {img_file}")  #unreadable files

    return data  #return  full list of image-label pairs

#prep data for model
def preprocess(data):
    X, y = [], []

    for img, label in data:
        X.append(img)
        y.append(label)

    #arrays + normalize
    X = np.array(X, dtype="float32").reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
    y_arr = np.array(y, dtype=int)  #0/1

    #split on raw labels first
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_arr, test_size=0.2, random_state=42, stratify=y_arr
    )

    #encode each split (to match Dense(2, softmax))
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
    y_val   = tf.keras.utils.to_categorical(y_val,   num_classes=2)

    return X_train, X_val, y_train, y_val

#build cnn
def build_model():
    model = Sequential([

        #first convolutional layer and max pooling
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        MaxPooling2D(2, 2),

        #second convolutional layer and max pooling
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        #flatten before dense layers
        Flatten(),

        #prevent overfitting
        Dropout(0.5),

        #dense layer learning patterns
        Dense(64, activation='relu'),

        #output layer w/ softmax for binary classification
        Dense(2, activation='softmax')
    ])

    #compile  model with optimizer, loss function, and metrics
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model  #return the compiled model

#main
if __name__ == "__main__":
    print("Loading data...")
    data = load_data()  #load all image data from folders
    print(f"Loaded {len(data)} images.")  #cnfirm number of images loaded

    print("Preprocessing...")
    X_train, X_val, y_train, y_val = preprocess(data)  #prepare data for training

    print("Building model...")
    model = build_model()  

    X_train = np.asarray(X_train, dtype="float32")
    X_val   = np.asarray(X_val,   dtype="float32")
    y_train = np.asarray(y_train, dtype="float32")
    y_val   = np.asarray(y_val,   dtype="float32")

    print("Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=9,              #number of training passes through the data                     #10 is one too many
        batch_size=30,          #number of samples processed before model updates weights
        validation_data=(X_val, y_val)  #data used to validate accuracy after each epoch
    )

    #save model to disk
    model.save("dat_scan_model.h5")
    print("Model saved.")

#visualizataion
plt.figure(figsize=(1, 4))

#accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

#loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()

