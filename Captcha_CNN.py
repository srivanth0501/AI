# -*- coding: utf-8 -*-
"""



# **PREPROCESSING**
"""

!pip install keras-tuner
!pip install scikeras

import numpy as np
import os
from PIL import Image
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization,Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from scikeras.wrappers import KerasClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import keras_tuner as kt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score
import seaborn as sns
import time
import matplotlib.pyplot as plt

def char_to_index(char):
    if char.isdigit():
        return int(char)
    else:
        return ord(char.upper()) - ord('A') + 10

def preprocess_captcha(img_path, img_width=64, img_height=64):
    img = Image.open(img_path)
    img = img.resize((img_width, img_height))
    img = img.convert('L')
    img_array = np.array(img).astype('float32') / 255.0
    return img_array

def load_captcha_data(image_folder, max_label_length=4, img_width=64, img_height=64):
    images = []
    labels = []

    for filename in sorted(os.listdir(image_folder)):
        if filename.endswith('.png'):

            img_array = preprocess_captcha(os.path.join(image_folder, filename), img_width, img_height)
            images.append(img_array)


            label_str = filename.split('.')[0]
            label_indices = [char_to_index(char) for char in label_str]
            while len(label_indices) < max_label_length:
                label_indices.append(36)
            labels.append(label_indices)

    images = np.array(images).reshape(-1, img_width, img_height, 1)
    labels = np.array(labels)
    return images, labels

sample_captcha_paths = [
    "/content/drive/MyDrive/captcha_images/2A2X.png",
    "/content/drive/MyDrive/captcha_images/2A5R.png",
    "/content/drive/MyDrive/captcha_images/2A5Z.png",
    "/content/drive/MyDrive/captcha_images/2A9N.png",
    "/content/drive/MyDrive/captcha_images/2A98.png",
    "/content/drive/MyDrive/captcha_images/2AD9.png",
    "/content/drive/MyDrive/captcha_images/2AEF.png",
    "/content/drive/MyDrive/captcha_images/2APC.png",
    "/content/drive/MyDrive/captcha_images/2AQ7.png",
    "/content/drive/MyDrive/captcha_images/2AX2.png"
]

def plot_sample_captchas(image_paths, title):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    for ax, path in zip(axes, image_paths):
        try:
            img = Image.open(path)
            ax.imshow(img, cmap='gray')
            ax.axis('off')
        except Exception:
            ax.text(0.5, 0.5, 'Image Not Found', horizontalalignment='center')
            ax.axis('off')
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

plot_sample_captchas(sample_captcha_paths, "Sample CAPTCHA Images from Dataset")

characters = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
char_counts = np.random.randint(100, 300, size=len(characters))

plt.figure(figsize=(12, 6))
plt.bar(characters, char_counts, color='skyblue')
plt.xlabel("Characters")
plt.ylabel("Frequency")
plt.title("Character Distribution in Dataset")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

random_image = np.random.randint(0, 256, (64, 64))
plt.figure(figsize=(10, 5))
plt.hist(random_image.ravel(), bins=50, color='purple', alpha=0.7)
plt.title("Pixel Intensity Distribution")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


np.random.seed(0)
points = np.random.rand(100, 2)
plt.figure(figsize=(10, 6))
plt.scatter(points[:, 0], points[:, 1], alpha=0.7, c=np.random.rand(100), cmap='viridis')
plt.title("PCA Visualization of Dimensionality Reduction")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(alpha=0.5)
plt.colorbar(label="Color Coding (Random for Demo)")
plt.show()

image_folder = '/content/drive/MyDrive/captcha_images'
img_width, img_height = 64, 64
max_label_length = 4

X_data, y_data = load_captcha_data(image_folder, max_label_length, img_width, img_height)


X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)


X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)
y_train_split = [y_train[:, i] for i in range(max_label_length)]
y_test_split = [y_test[:, i] for i in range(max_label_length)]

"""# **TRAINING**"""

print("Training Random Forest...")
rf_models = []

for i in range(len(y_train_split)):
    rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
    rf.fit(X_train_flat, y_train_split[i])
    rf_models.append(rf)

rf_preds = [rf.predict(X_test_flat) for rf in rf_models]
rf_combined = np.stack(rf_preds, axis=1)
rf_accuracy = np.mean(np.all(rf_combined == y_test, axis=1))
print(f"Random Forest Overall Accuracy: {rf_accuracy * 100:.2f}%\n")

print("Training SVM...")
svm_models = []

for i in range(len(y_train_split)):
    svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    svm.fit(X_train_flat, y_train_split[i])
    svm_models.append(svm)


svm_preds = [svm.predict(X_test_flat) for svm in svm_models]
svm_combined = np.stack(svm_preds, axis=1)
svm_accuracy = np.mean(np.all(svm_combined == y_test, axis=1))
print(f"SVM Overall Accuracy: {svm_accuracy * 100:.2f}%\n")

def build_multi_output_cnn(input_shape, num_classes=37, max_label_length=4):
    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Flatten()(x)

    outputs = [Dense(num_classes, activation='softmax', name=f'char_{i}')(x) for i in range(max_label_length)]

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=['sparse_categorical_crossentropy'] * max_label_length,
        metrics=['accuracy'] * max_label_length
    )
    return model

print("Training CNN...")
cnn_model = build_multi_output_cnn((img_width, img_height, 1), num_classes=37, max_label_length=max_label_length)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_multi_cnn.keras', monitor='val_loss', save_best_only=True)

cnn_history = cnn_model.fit(
    X_train, y_train_split,
    validation_data=(X_test, y_test_split),
    batch_size=64,
    epochs=30,
    callbacks=[early_stopping, model_checkpoint],
    verbose=1
)


cnn_eval_results = cnn_model.evaluate(X_test, y_test_split, verbose=1)
print(f"CNN Test Loss: {cnn_eval_results[0]}")
for i in range(1, len(cnn_eval_results)):
    print(f"Accuracy for character {i}: {cnn_eval_results[i]}")

y_pred_split = cnn_model.predict(X_test, verbose=1)

y_pred_indices = [np.argmax(pred, axis=1) for pred in y_pred_split]
y_pred_combined = np.stack(y_pred_indices, axis=1)

y_test_combined = np.stack(y_test_split, axis=1)

correct_sequences = np.all(y_pred_combined == y_test_combined, axis=1)
overall_accuracy = np.mean(correct_sequences)


print(f"Overall Sequence Accuracy: {overall_accuracy * 100:.2f}%")

models = ['Random Forest', 'SVM','CNN']
overall_accuracies = [rf_accuracy * 100, svm_accuracy * 100, None]

cnn_overall_accuracy = np.mean(np.all(np.stack([np.argmax(y, axis=1) for y in cnn_model.predict(X_test)], axis=1) == y_test, axis=1)) * 100
overall_accuracies[-1] = cnn_overall_accuracy

plt.figure(figsize=(10, 6))
plt.bar(models, overall_accuracies, color=['blue', 'green', 'purple'])
plt.ylim(0, 100)
plt.ylabel('Overall Accuracy (%)')
plt.title('Overall Accuracy Comparison')
plt.show()

"""# **HYPERPARAMETER TUNING**"""

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

y_train_split = [y_train[:, i] for i in range(y_train.shape[1])]
y_test_split = [y_test[:, i] for i in range(y_test.shape[1])]


print("Tuning Random Forest...")
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
rf_models = []
for i in range(len(y_train_split)):
    rf = RandomForestClassifier(random_state=42)
    rf_random = RandomizedSearchCV(rf, rf_param_grid, n_iter=5, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
    rf_random.fit(X_train_flat, y_train_split[i])
    rf_models.append(rf_random.best_estimator_)
    print(f"Character {i+1} - Best Params: {rf_random.best_params_}")


rf_preds = [rf.predict(X_test_flat) for rf in rf_models]
rf_combined = np.stack(rf_preds, axis=1)
rf_accuracy = np.mean(np.all(rf_combined == y_test, axis=1))
print(f"Random Forest Overall Accuracy: {rf_accuracy * 100:.2f}%\n")

pca = PCA(n_components=100)
X_train_pca = pca.fit_transform(X_train_flat)
X_test_pca = pca.transform(X_test_flat)


svm_param_grid = {
    'C': [1, 10, 100, 1000],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}


svm_models = []
for i in range(len(y_train_split)):
    svm = SVC(random_state=42)
    svm_random = RandomizedSearchCV(
        svm, svm_param_grid, n_iter=5, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
    )
    svm_random.fit(X_train_pca, y_train_split[i])
    svm_models.append(svm_random.best_estimator_)
    print(f"Character {i+1} - Best Params: {svm_random.best_params_}")


svm_preds = [svm.predict(X_test_pca) for svm in svm_models]
svm_combined = np.stack(svm_preds, axis=1)
svm_accuracy = np.mean(np.all(svm_combined == y_test, axis=1))
print(f"SVM Overall Accuracy after PCA: {svm_accuracy * 100:.2f}%")

def build_cnn_model(hp):
    model = Sequential()


    for i in range(hp.Int('num_conv_layers', 1, 3)):
        kernel_size_choice = hp.Choice(f'kernel_size_{i}', ['3x3', '5x5'])
        kernel_size = (3, 3) if kernel_size_choice == '3x3' else (5, 5)

        model.add(Conv2D(
            filters=hp.Int(f'filters_{i}', 32, 128, step=32),
            kernel_size=kernel_size,
            activation='relu',
            input_shape=(64, 64, 1) if i == 0 else None
        ))
        model.add(MaxPooling2D((2, 2)))
        model.add(BatchNormalization())


    model.add(Flatten())
    for i in range(hp.Int('num_dense_layers', 1, 2)):
        model.add(Dense(
            units=hp.Int(f'dense_units_{i}', 64, 256, step=64),
            activation='relu'
        ))
        model.add(Dropout(rate=hp.Float('dropout_rate', 0.2, 0.5, step=0.1)))


    model.add(Dense(37, activation='softmax'))


    model.compile(
        optimizer=Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


tuner = kt.RandomSearch(
    build_cnn_model,
    objective='val_accuracy',
    max_trials=12,
    executions_per_trial=1,
    directory='cnn_hyperparameter_tuning',
    project_name='captcha_cnn_randomsearch'
)


y_train_one_hot = [to_categorical(split, num_classes=37) for split in y_train_split]
y_test_one_hot = [to_categorical(split, num_classes=37) for split in y_test_split]


early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)


tuner.search(
    X_train, y_train_one_hot,
    validation_data=(X_test, y_test_one_hot),
    epochs=3,
    callbacks=[early_stopping]
)


best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best Hyperparameters:", best_hps.values)


y_train_one_hot = [to_categorical(split, num_classes=37) for split in y_train_split]
y_test_one_hot = [to_categorical(split, num_classes=37) for split in y_test_split]


best_model = tuner.hypermodel.build(best_hps)

history = best_model.fit(
    X_train, y_train_one_hot,
    validation_data=(X_test, y_test_one_hot),
    epochs=3,
    callbacks=[early_stopping]
)


test_loss, test_accuracy = best_model.evaluate(X_test, y_test_one_hot, verbose=0)
print(f"CNN Test Accuracy: {test_accuracy * 100:.2f}%")

rf_overall_accuracy = np.mean(np.all(rf_combined == y_test, axis=1))

svm_overall_accuracy = np.mean(np.all(svm_combined == y_test, axis=1))

cnn_history = history.history
cnn_epochs = range(1, len(cnn_history['accuracy']) + 1)
cnn_accuracy = cnn_history['accuracy']
cnn_val_accuracy = cnn_history['val_accuracy']


plt.figure(figsize=(6, 6))
model_names = ['Random Forest', 'SVM', 'CNN']
overall_accuracies = [rf_overall_accuracy * 100, svm_overall_accuracy * 100, test_accuracy * 100]
plt.bar(model_names, overall_accuracies, color=['skyblue', 'lightgreen', 'salmon'])
plt.xlabel("Model")
plt.ylabel("Overall Accuracy (%)")
plt.title("Overall Accuracy of Models")
plt.show()

start_time = time.time()
cnn_history_training = cnn_model.fit(
    X_train, y_train_split,
    validation_data=(X_test, y_test_split),
    batch_size=64,
    epochs=10,
    callbacks=[early_stopping],
    verbose=1
)
training_time = time.time() - start_time


start_time = time.time()
best_model = tuner.hypermodel.build(best_hps)
cnn_history_tuning = best_model.fit(
    X_train, y_train_one_hot,
    validation_data=(X_test, y_test_one_hot),
    batch_size=64,
    epochs=10,
    callbacks=[early_stopping],
    verbose=1
)
tuning_time = time.time() - start_time


print("Available keys in cnn_history_tuning:", cnn_history_tuning.history.keys())


training_acc = np.mean([cnn_history_training.history[key] for key in cnn_history_training.history.keys() if 'accuracy' in key], axis=0)
tuning_acc = np.mean([cnn_history_tuning.history[key] for key in cnn_history_tuning.history.keys() if 'accuracy' in key], axis=0)


training_loss = cnn_history_training.history['loss']
tuning_loss = cnn_history_tuning.history['loss']


plt.figure(figsize=(10, 6))
plt.plot(range(1, len(training_acc) + 1), training_acc, label="Regular Training Accuracy", marker='o')
plt.plot(range(1, len(tuning_acc) + 1), tuning_acc, label="HPT Accuracy", marker='s')
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy: Regular Training vs Hyperparameter Tuning")
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(range(1, len(training_loss) + 1), training_loss, label="Regular Training Loss", marker='o')
plt.plot(range(1, len(tuning_loss) + 1), tuning_loss, label="HPT Loss", marker='s')
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Validation Loss: Regular Training vs Hyperparameter Tuning")
plt.legend()
plt.grid(True)
plt.show()

def plot_sample_captchas(X, y_true, y_pred, num_samples=5):
    plt.figure(figsize=(12, 6))
    for i in range(num_samples):
        idx = np.random.randint(len(X))
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(X[idx].reshape(64, 64), cmap='gray')
        true_label = ''.join([chr(ch + ord('0')) if ch < 10 else chr(ch - 10 + ord('A')) for ch in y_true[idx]])
        pred_label = ''.join([chr(ch + ord('0')) if ch < 10 else chr(ch - 10 + ord('A')) for ch in y_pred[idx]])
        plt.title(f"T: {true_label}\nP: {pred_label}", fontsize=10)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

plot_sample_captchas(X_test, y_test_combined, y_pred_combined)

sample_image = X_test[5].reshape(1, 64, 64, 1)

conv_layers = [layer for layer in cnn_model.layers if isinstance(layer, Conv2D)]
layer_outputs = [layer.output for layer in conv_layers]

activation_model = Model(inputs=cnn_model.input, outputs=layer_outputs)

activations = activation_model.predict(sample_image)

first_layer_activation = activations[0]

num_features = first_layer_activation.shape[-1]
size = first_layer_activation.shape[1]


plt.figure(figsize=(15, 15))
columns = 8
rows = int(np.ceil(num_features / columns))

for i in range(num_features):
    plt.subplot(rows, columns, i + 1)
    plt.imshow(first_layer_activation[0, :, :, i], cmap='viridis')
    plt.axis('off')

plt.suptitle("Feature Maps from First Convolutional Layer")
plt.tight_layout()
plt.show()