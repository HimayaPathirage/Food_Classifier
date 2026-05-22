# Import Libraries
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers

# ---------------- CONFIG ----------------

IMG_SIZE = 160
ROOT_PATH = r"C:\Users\Chathurika Dinushi\Desktop\Final_AI_Project\FOOD"
NUTRITION_CSV = r"C:\Users\Chathurika Dinushi\Desktop\Final_AI_Project\custom_nutrition_data.csv"
SAVE_DIR = r"C:\Users\Chathurika Dinushi\Desktop\Final_AI_Project\outputs_high_accuracy"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------- LOAD IMAGES ----------------


def load_images_from_folder(root_folder, class_names, label_map):
    X_data, y_data = [], []
    for class_name in class_names:
        folder_path = os.path.join(root_folder, class_name)
        for file in os.listdir(folder_path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(folder_path, file)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    img = img.astype('float32') / 255.0  # Normalize to [0, 1]
                    X_data.append(img)
                    y_data.append(label_map[class_name])
    return np.array(X_data), np.array(y_data)


class_names = sorted(os.listdir(os.path.join(ROOT_PATH, 'training')))
label_map = {name: idx for idx, name in enumerate(class_names)}
num_classes = len(class_names)

X_train, y_train_labels = load_images_from_folder(
    os.path.join(ROOT_PATH, 'training'), class_names, label_map)
X_val, y_val_labels = load_images_from_folder(
    os.path.join(ROOT_PATH, 'validation'), class_names, label_map)
X_test, y_test_labels = load_images_from_folder(
    os.path.join(ROOT_PATH, 'evaluation'), class_names, label_map)

# Standardization
mean = np.mean(X_train, axis=(0, 1, 2))
std = np.std(X_train, axis=(0, 1, 2))
X_train = (X_train - mean) / (std + 1e-7)
X_val = (X_val - mean) / (std + 1e-7)
X_test = (X_test - mean) / (std + 1e-7)

# One-hot labels
y_train = to_categorical(y_train_labels, num_classes=num_classes)
y_val = to_categorical(y_val_labels, num_classes=num_classes)
y_test = to_categorical(y_test_labels, num_classes=num_classes)

# ---------------- DATA AUGMENTATION ----------------#
datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    shear_range=0.15,
    fill_mode='nearest'
)
datagen.fit(X_train)

# Class weights
weights = compute_class_weight(
    class_weight='balanced', classes=np.unique(y_train_labels), y=y_train_labels)
class_weights = dict(enumerate(weights))

# ---------------- STRONGER MODEL ARCHITECTURE ----------------

l2_reg = 1e-4  # L2 regularization factor

model = Sequential([
    # Input Block
    Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2_reg),
           input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(64, (3, 3), padding='same',
           kernel_regularizer=regularizers.l2(l2_reg)),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),

    # Block 2
    Conv2D(128, (3, 3), padding='same',
           kernel_regularizer=regularizers.l2(l2_reg)),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(128, (3, 3), padding='same',
           kernel_regularizer=regularizers.l2(l2_reg)),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.4),

    # Block 3
    Conv2D(256, (3, 3), padding='same',
           kernel_regularizer=regularizers.l2(l2_reg)),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(256, (3, 3), padding='same',
           kernel_regularizer=regularizers.l2(l2_reg)),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.4),

    # Block 4
    Conv2D(512, (3, 3), padding='same',
           kernel_regularizer=regularizers.l2(l2_reg)),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(512, (3, 3), padding='same',
           kernel_regularizer=regularizers.l2(l2_reg)),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.5),

    # Classification Head
    GlobalAveragePooling2D(),
    Dense(512, kernel_regularizer=regularizers.l2(l2_reg)),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])


model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ---------------- CALLBACKS ----------------

early_stop = EarlyStopping(
    monitor='val_loss', patience=15, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss', patience=6, factor=0.5, verbose=1)
checkpoint = ModelCheckpoint(os.path.join(SAVE_DIR, "best_food_model_v2.h5"),
                             monitor='val_accuracy', save_best_only=True, verbose=1)
csv_logger = CSVLogger(os.path.join(
    SAVE_DIR, "training_log_v2.csv"), append=True)

# ---------------- TRAIN ----------------
history = model.fit(

    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=150,
    class_weight=class_weights,
    callbacks=[early_stop, lr_scheduler, checkpoint, csv_logger]
)


# --- PLOTS ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(os.path.join(SAVE_DIR, "training_curves_v2.png"))
plt.show()

# --- EVALUATION ---
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)

report = classification_report(
    y_test_labels, y_pred_labels, target_names=class_names)
print(report)

with open(os.path.join(SAVE_DIR, "evaluation_report_v2.txt"), "w") as f:
    f.write(report)

cm = confusion_matrix(y_test_labels, y_pred_labels)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names,
            yticklabels=class_names, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix_v2.png"))
plt.show()

# --- SAVE FINAL MODEL ---
model.save(os.path.join(SAVE_DIR, "final_food_model_v2.h5"))
print("Final model saved successfully!")


# ---------------- NUTRITION LOOKUP----------------

nutrition_df = pd.read_csv(NUTRITION_CSV)
nutrition_df['Food_Item'] = nutrition_df['Food_Item'].str.strip().str.lower()


def get_nutrition(food_name):
    name = food_name.strip().lower()
    match = nutrition_df[nutrition_df['Food_Item'] == name]
    if not match.empty:
        return match.iloc[0][['Calories', 'Carbs', 'Protein', 'Fat']].to_dict()
    return {'Calories': 'N/A', 'Carbs': 'N/A', 'Protein': 'N/A', 'Fat': 'N/A'}


def print_true_vs_pred(index):

    img_standardized = X_test[index]
    img_exp = np.expand_dims(img_standardized, axis=0)

    prediction = model.predict(img_exp)
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = class_names[predicted_class_index]

    true_class_index = y_test_labels[index]
    true_class_name = class_names[true_class_index]

    print(f'\n--- Prediction for Test Image Index: {index} ---')
    print(f'   True Class:     {true_class_name}')
    print(f'   Predicted Class:  {predicted_class_name}')

    nutrition = get_nutrition(predicted_class_name)
    print('\n   Nutrition Information (for predicted class):')
    for k, v in nutrition.items():
        print(f'     - {k}: {v}')
    print('-------------------------------------------')


# Example usage for 5 random images
for i in np.random.choice(len(X_test), 5, replace=False):
    print_true_vs_pred(i)
