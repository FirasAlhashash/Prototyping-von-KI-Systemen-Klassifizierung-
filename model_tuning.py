from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam, RMSprop, SGD
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import KFold
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import tensorflow as tf
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.callbacks import TensorBoard

IMAGE_SIZE = (32, 32)
NUM_CLASSES = 2

image_directory = r'C:\Users\kolos\OneDrive\Desktop\Uni Aufgaben\Prototyping Projekt\preview2'
image_v_directory = r'C:\Users\kolos\OneDrive\Desktop\Uni Aufgaben\Prototyping Projekt\validations_preview'
images = []
labels = []
images_v = []
labels_v = []
for filename in os.listdir(image_directory):
    if filename.endswith('.jpeg'):
        img_path = os.path.join(image_directory, filename)
        img = load_img(img_path, target_size=IMAGE_SIZE)
        x = img_to_array(img)
        images.append(x)
        if 'True' in filename:
            labels.append(1)  # True label
        else:
            labels.append(0)  # False label

for filename in os.listdir(image_v_directory):
    if filename.endswith('.jpeg'):
        img_path = os.path.join(image_v_directory, filename)
        img = load_img(img_path, target_size=IMAGE_SIZE)
        x = img_to_array(img)
        images_v.append(x)
        if 'True' in filename:
            labels_v.append(1)  # True label
        else:
            labels_v.append(0)  # False label
images_v = np.array(images) / 255.0
labels_v = np.array(labels)

def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
seed = 7
tf.random.set_seed(seed)
# Define the parameter grid as usual
model = KerasClassifier(model=create_model, verbose=0)
batch_size = [5, 10]
epochs = [5, 10]

tensorboard_callback = TensorBoard(log_dir='tensorboard_log_dic', histogram_freq=1)
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(images, labels, validation_data=(images_v, labels_v), callbacks=[tensorboard_callback])
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))