import os
import pickle
from img2vec_pytorch import Img2Vec
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
img2vec = Img2Vec()
data_dir = '/data/share/public/handaway/train4'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')

data = {}
for j, dir_ in enumerate([train_dir, val_dir]):
    features = []
    labels = []
    for category in os.listdir(dir_):
        for img_path in os.listdir(os.path.join(dir_, category)):
            img_path_ = os.path.join(dir_, category, img_path)
            img = Image.open(img_path_)

            img_features = img2vec.get_vec(img)

            features.append(img_features)
            labels.append(category)

    data[['training_data', 'validation_data'][j]] = features
    data[['training_labels', 'validation_labels'][j]] = labels

from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)  # Keep components that explain 95% of the variance

# Fit and transform the training features
pca.fit(data['training_data'])
transformed_training_data = pca.transform(data['training_data'])

# Transform the validation features using the fitted PCA
transformed_validation_data = pca.transform(data['validation_data'])

# Replace the original data with the transformed data
data['training_data'] = transformed_training_data
data['validation_data'] = transformed_validation_data
model = RandomForestClassifier(random_state=0)
model.fit(data['training_data'], data['training_labels'])

# test performance
y_pred = model.predict(data['validation_data'])
score = accuracy_score(y_pred, data['validation_labels'])

print(score)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Create a KNN model with k=5 (you can adjust this hyperparameter)
model = KNeighborsClassifier(n_neighbors=3)

# Fit the model to the training data
model.fit(data['training_data'], data['training_labels'])

# Test performance on the validation data
y_pred = model.predict(data['validation_data'])
score = accuracy_score(y_pred, data['validation_labels'])

print(score)