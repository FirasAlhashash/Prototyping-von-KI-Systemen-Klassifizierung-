import os
import pickle
from img2vec_pytorch import Img2Vec
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
from sklearn.decomposition import PCA
import numpy as np

img2vec = Img2Vec()
data_dir = r'C:\Projekt\train4'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
test_dir = r'C:\Projekt\test4'

data = {}

# Funktion zum Extrahieren der Kategorie aus dem Dateinamen
def get_category(filename):
    return filename.split('_')[1].split('.')[0]  # Annahme: Format ist "number_category.jpg"

for j, dir_ in enumerate([train_dir, val_dir]):
    features = []
    labels = []
    for category in os.listdir(dir_):
        category_path = os.path.join(dir_, category)
        if os.path.isdir(category_path):
            for img_path in os.listdir(category_path):
                img_path_ = os.path.join(category_path, img_path)
                img = Image.open(img_path_)
                img_features = img2vec.get_vec(img)
                features.append(img_features)
                labels.append(category)
    data[['training_data', 'validation_data'][j]] = features
    data[['training_labels', 'validation_labels'][j]] = labels

# Separate Verarbeitung für Testdaten
test_features = []
test_labels = []
for img_path in os.listdir(test_dir):
    if img_path.endswith('.jpg') or img_path.endswith('.png'):  # Nur Bilddateien verarbeiten
        img_path_ = os.path.join(test_dir, img_path)
        img = Image.open(img_path_)
        img_features = img2vec.get_vec(img)
        test_features.append(img_features)
        test_labels.append(get_category(img_path))

data['test_data'] = test_features
data['test_labels'] = test_labels

# Rest des Codes bleibt unverändert
# PCA
pca = PCA(n_components=0.95)
pca.fit(data['training_data'])
data['training_data'] = pca.transform(data['training_data'])
data['validation_data'] = pca.transform(data['validation_data'])
data['test_data'] = pca.transform(data['test_data'])

# Grid Search
param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}

knn = KNeighborsClassifier()

grid_search = GridSearchCV(
    knn, 
    param_grid, 
    cv=5, 
    scoring=['accuracy', 'precision_macro', 'f1_macro'],
    refit='accuracy',
    n_jobs=-1
)

grid_search.fit(data['training_data'], data['training_labels'])

# Bestes Modell
best_model = grid_search.best_estimator_

# Vorhersagen mit dem besten Modell auf Testdaten
y_pred = best_model.predict(data['test_data'])

# Metriken berechnen für Testdaten
accuracy = accuracy_score(data['test_labels'], y_pred)
precision = precision_score(data['test_labels'], y_pred, average='macro')
f1 = f1_score(data['test_labels'], y_pred, average='macro')

print(f"Best parameters: {grid_search.best_params_}")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test F1 Score: {f1:.4f}")

# Confusion Matrix für Testdaten
conf_matrix = confusion_matrix(data['test_labels'], y_pred)
print("Confusion Matrix (Test Data):")
print(conf_matrix)

# Visualisierung der Confusion Matrix
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Test Data)')
plt.show()
