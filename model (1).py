
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import seaborn as sns
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

IMAGE_SIZE = (32, 32)
NUM_CLASSES = 2
BATCH_SIZE = 10
EPOCHS = 3


image_directory = r'C:\Users\kolos\OneDrive\Desktop\Uni Aufgaben\Prototyping Projekt\preview'
images = []
labels = []

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

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Vorbereiten der Daten für Cross-Validation
images = np.array(images) / 255.0
labels = np.array(labels)

# Speichern der Ergebnisse
fold_no = 1
loss_per_fold = []
accuracy_per_fold = []
model_architecture = Sequential([
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
model_architecture.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

for train_index, val_index in kf.split(images):
    X_train, X_val = images[train_index], images[val_index]
    y_train, y_val = labels[train_index], labels[val_index]
    
    model = model_architecture
    # Trainiere das Modell mit dem aktuellen Fold
    print(f'Training for fold {fold_no} ...')
    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, 
                        validation_data=(X_val, y_val), verbose=1)
    
    # Evaluiere das Modell
    scores = model.evaluate(X_val, y_val, verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    loss_per_fold.append(scores[0])
    accuracy_per_fold.append(scores[1] * 100)
    
    # Nächste Fold
    fold_no += 1

# Durchschnittliche Ergebnisse aus allen Folds

print('Score per fold')
for i in range(0, len(accuracy_per_fold)):
    print('------------------------------------------------------------------------')
    print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {accuracy_per_fold[i]}%')

print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(accuracy_per_fold)} (+- {np.std(accuracy_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')


# Pfad zum Verzeichnis mit den Testdaten
test_data_directory = r'C:\Users\kolos\OneDrive\Desktop\Uni Aufgaben\Prototyping Projekt\test_daten'


model = model_architecture


model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# trainiern
model.fit(images, labels, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)
# Laden der Testdaten
test_images = []
test_labels = []

for filename in os.listdir(test_data_directory):
    if filename.endswith('.jpeg'):
        img_path = os.path.join(test_data_directory, filename)
        img = load_img(img_path, target_size=IMAGE_SIZE)
        x = img_to_array(img)
        test_images.append(x)
        if 'True' in filename:
            test_labels.append(1)  # True label
        else:
            test_labels.append(0)  # False label

            
test_images = np.array(test_images) / 255.0
test_labels = np.array(test_labels)

predictions = model.predict(test_images)
predictions = [1 if pred > 0.5 else 0 for pred in predictions]

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_labels, predictions)
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Konfusionsmatrix')
plt.ylabel('Tatsächliche Klasse')
plt.xlabel('Vorhergesagte Klasse')
plt.show()


TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]

print(f'True Positives: {TP}')
print(f'True Negatives: {TN}')
print(f'False Positives: {FP}')
print(f'False Negatives: {FN}')