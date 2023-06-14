import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.applications.inception_v3 import InceptionV3
from keras.utils import to_categorical, load_img, img_to_array

# Define the list of disease names and create a dictionary to map labels to numbers
disease_names = ['Acne and Rosacea Photos', 'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions', 'Atopic Dermatitis Photos', 'Bullous Disease Photos', 'Eczema Photos','Melanoma Skin Cancer Nevi and Moles', 'Nail Fungus and other Nail Disease', 'Systemic Disease', 'Urticaria Hives', 'Vascular Tumors']
label_dict = {disease_names[i]: i for i in range(len(disease_names))}

# Load the images and labels from the photos folder
images = []
labels = []
for disease_name in disease_names:
    folder_path = os.path.join('Dataset', disease_name)
    for filename in os.listdir(folder_path):         
        image_path = os.path.join(folder_path, filename)
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)
        images.append(image)
        labels.append(label_dict[disease_name])

# Convert the data to arrays and normalize the pixel values
X = np.array(images) / 255.0
y = np.array(labels)

# Convert the labels to categorical values
y = to_categorical(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Load the pre-trained InceptionV3 model
base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Create the CNN model by adding layers on top of the base model
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(disease_names), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# Save the trained model
model.save('skin_disease_inceptionv3.h5')

# Use the model to predict the image 'trial.jpg'
image_path = 'trial.jpg'
image = load_img(image_path, target_size=(224, 224))
image = img_to_array(image)
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
image = image / 255.0
prediction = model.predict(image)
predicted_disease = disease_names[np.argmax(prediction)]
print('The detected skin disease is:', predicted_disease)
