
import os
import shutil
import numpy as np
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
IMAGE_SIZE = (224, 224)
CATEGORIES = ["Personal", "Work", "Notes", "People"]
DATASET_PATH = "dataset"
MODEL_PATH = "image_classifier_model_m2.h5"
def load_dataset():
    images, labels = [], []
    for label, category in enumerate(CATEGORIES):
        category_path = os.path.join(DATASET_PATH, category)
        for image_file in os.listdir(category_path):
            image_path = os.path.join(category_path, image_file)
            try:
                img = load_img(image_path, target_size=IMAGE_SIZE)
                img_array = img_to_array(img) / 255.0  # Normalize
                images.append(img_array)
                labels.append(label)
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
    return np.array(images), np.array(labels)
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(CATEGORIES), activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
def train_model():
    images, labels = load_dataset()
    labels = to_categorical(labels, num_classes=len(CATEGORIES))
    train_images, val_images, train_labels, val_labels = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )
    datagen = ImageDataGenerator(
        rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
        zoom_range=0.2, horizontal_flip=True, fill_mode='nearest'
    )
    datagen.fit(train_images)
    model = build_model()
    model.fit(
        datagen.flow(train_images, train_labels, batch_size=32),
        validation_data=(val_images, val_labels),
        epochs=50
    )
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
def predict_category(image_path, model):
    img = load_img(image_path, target_size=IMAGE_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    return CATEGORIES[predicted_index]
def categorize_images(input_folder):
    model = load_model(MODEL_PATH)
    for image_file in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_file)
        if not os.path.isfile(image_path):
            continue

        category = predict_category(image_path, model)
        output_folder = os.path.join(category)
        os.makedirs(output_folder, exist_ok=True)
        shutil.move(image_path, os.path.join(output_folder, image_file))
        print(f"Moved {image_file} to {category}/")

if __name__ == "__main__":
   #train_model()
   input_folder = "input_images"
   categorize_images(input_folder)