from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import confusion_matrix
import numpy as np
import os

# Définir les chemins pour le dataset
base_dir = '/Users/usmb/Desktop/Modélisation et réseaux de neurones formels/Combined Dataset'
train_dir = os.path.join(base_dir, 'train')  # Répertoire d'entraînement
test_dir = os.path.join(base_dir, 'test')   # Répertoire de test

# Générateurs d'images pour les données d'entraînement et de test
train_generator = ImageDataGenerator(
    rescale=1./255,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True
)

test_generator = ImageDataGenerator(
    rescale=1./255
)

# Charger les données d'entraînement
train_images = train_generator.flow_from_directory(
    train_dir,
    target_size=(64, 64),  # Correspond à l'input_shape du modèle
    batch_size=32,
    class_mode='categorical'
)

# Charger les données de test
test_images = test_generator.flow_from_directory(
    test_dir,
    target_size=(64, 64),  # Correspond à l'input_shape du modèle
    batch_size=32,
    class_mode='categorical'
)

# Construire le modèle CNN
network = Sequential()

network.add(Conv2D(filters=32, kernel_size=3, strides=1, input_shape=(64, 64, 3), activation='relu'))
network.add(MaxPooling2D(pool_size=(2, 2)))
network.add(Dropout(0.2))

network.add(Conv2D(filters=64, kernel_size=3, strides=1, activation='relu'))
network.add(MaxPooling2D(pool_size=(2, 2)))
network.add(Dropout(0.3))

network.add(Conv2D(filters=128, kernel_size=3, strides=1, activation='relu'))
network.add(MaxPooling2D(pool_size=(2, 2)))
network.add(Dropout(0.3))

network.add(Flatten())
network.add(Dense(units=64, activation='relu'))
network.add(Dropout(0.3))
network.add(Dense(units=32, activation='relu'))
network.add(Dense(units=4, activation='softmax'))

# Compilation du modèle
network.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Entraîner le modèle
print("Phase 1 : Entraînement initial...")
history = network.fit(train_images, epochs=10, validation_data=test_images)

# Déverrouiller certaines couches pour le fine-tuning
print("Phase 2 : Fine-tuning...")
for layer in network.layers[:5]:  # Par exemple, déverrouiller les 5 premières couches
    layer.trainable = True

# Recompiler le modèle avec un faible taux d'apprentissage
from tensorflow.keras.optimizers import Adam
network.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Entraîner le modèle sur les données d'entraînement avec fine-tuning
history_finetune = network.fit(train_images, epochs=20, validation_data=test_images)

# Sauvegarder le modèle
network.save("alzheimer_finetuned_cnn")

# Évaluer le modèle sur les données de test
steps = test_images.samples // test_images.batch_size + int(test_images.samples % test_images.batch_size != 0)
predictions = network.predict(test_images, steps=steps)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_images.classes

# Vérifier les longueurs avant de calculer la matrice de confusion
if len(predicted_classes) == len(true_classes):
    cm = confusion_matrix(true_classes, predicted_classes)
    print("Matrice de confusion :\n", cm)
else:
    print("Erreur : Les dimensions des prédictions et des classes réelles ne correspondent pas.")

# Afficher les indices de classes
print("Classes :", test_images.class_indices)

# Fonction pour tester une image unique
def test_image(path):
    """
    Fonction pour tester une seule image et prédire le stade de la maladie d'Alzheimer.
    :param path: Chemin de l'image à tester.
    :return: Résultat de la prédiction sous forme de texte.
    """
    # Charger l'image et la redimensionner
    tmp_image = load_img(path, target_size=(64, 64))
    tmp_image = img_to_array(tmp_image)  # Convertir l'image en tableau numpy
    tmp_image = np.expand_dims(tmp_image, axis=0)  # Ajouter une dimension pour correspondre au batch
    
    # Faire une prédiction
    result = network.predict(tmp_image)
    
    # Trouver la classe prédite (indice de la classe avec la plus grande probabilité)
    predicted_class = np.argmax(result, axis=1)[0]
    
    # Traduire la classe prédite en un texte lisible
    if predicted_class == 3:
        return "Alzheimer très léger"
    elif predicted_class == 0:
        return "Alzheimer léger"
    elif predicted_class == 1:
        return "Alzheimer modéré"
    else:
        return "Aucun Alzheimer"

# Exemple d'utilisation
image_path = '/Users/usmb/Desktop/Modélisation et réseaux de neurones formels/Combined Dataset/test/Moderate Impairment/27 (2).jpg' # Remplacez par le chemin de votre image de test
result = test_image(image_path)
print("Résultat de la prédiction :", result)
