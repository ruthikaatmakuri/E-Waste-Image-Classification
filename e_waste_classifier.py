# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import gradio as gr
from PIL import Image

# -----------------------------
# Load Dataset Paths
# -----------------------------
testpath = r'C:/Users/parth/Downloads/E-Waste classification dataset/modified-dataset/test'
trainpath = r'C:/Users/parth/Downloads/E-Waste classification dataset/modified-dataset/train'
validpath = r'C:/Users/parth/Downloads/E-Waste classification dataset/modified-dataset/val'

# -----------------------------
# Load image datasets
# -----------------------------
# Datasets are split into training, validation, and testing
datatrain = tf.keras.utils.image_dataset_from_directory(trainpath, shuffle=True, image_size=(128, 128), batch_size=32)
datatest = tf.keras.utils.image_dataset_from_directory(testpath, shuffle=False, image_size=(128, 128), batch_size=32)
datavalid = tf.keras.utils.image_dataset_from_directory(validpath, shuffle=True, image_size=(128, 128), batch_size=32)

# Get class labels from dataset
class_names = datatrain.class_names

# -----------------------------
# Visualize a batch of training images
# -----------------------------
plt.figure(figsize=(10, 10))
for images, labels in datatrain.take(1):
    for i in range(12):
        ax = plt.subplot(4, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()

# -----------------------------
# Function to plot class distribution
# -----------------------------
def plot_class_distribution(dataset, title="Class Distribution"):
    class_counts = {}
    for images, labels in dataset:
        for label in labels.numpy():
            class_name = dataset.class_names[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    plt.figure(figsize=(10, 6))
    plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
    plt.xlabel("Class")
    plt.ylabel("Number of Items")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Visualize class distributions for each split
plot_class_distribution(datatrain, "Training Data Distribution")
plot_class_distribution(datavalid, "Validation Data Distribution")
plot_class_distribution(datatest, "Test Data Distribution")

# -----------------------------
# Data Augmentation Block
# -----------------------------
# Improves generalization by applying random transformations
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# -----------------------------
# Model Setup using Transfer Learning
# -----------------------------
# Load EfficientNetV2B0 with pre-trained weights from ImageNet
base_model = EfficientNetV2B0(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
base_model.trainable = True

# Freeze first 100 layers to retain learned features
for layer in base_model.layers[:100]:
    layer.trainable = False

# Build the classification model
model = Sequential([
    layers.Input(shape=(128, 128, 3)),
    data_augmentation,
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')  # 10 classes for prediction
])

# Compile the model with Adam optimizer and Sparse Categorical Loss
model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# -----------------------------
# Train the Model
# -----------------------------
early = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(datatrain, validation_data=datavalid, epochs=15, batch_size=100, callbacks=[early])

# -----------------------------
# Plot Accuracy and Loss Curves
# -----------------------------
acc = history.history.get('accuracy')
val_acc = history.history.get('val_accuracy')
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training vs Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training vs Validation Loss')
plt.show()

# -----------------------------
# Evaluate on Test Data
# -----------------------------
loss, accuracy = model.evaluate(datatest)
print(f'Test accuracy is {accuracy:.4f}, Test loss is {loss:.4f}')

# -----------------------------
# Generate Predictions and Confusion Matrix
# -----------------------------
y_true = np.concatenate([y.numpy() for x, y in datatest], axis=0)
y_pred_probs = model.predict(datatest)
y_pred = np.argmax(y_pred_probs, axis=1)

# Print raw confusion matrix and classification report
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))

# Plot confusion matrix heatmap
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# -----------------------------
# Show sample predictions from test data
# -----------------------------
for images, labels in datatest.take(1):
    predictions = model.predict(images)
    pred_labels = tf.argmax(predictions, axis=1)
    for i in range(8):
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(f"True: {class_names[labels[i]]}, Pred: {class_names[pred_labels[i]]}")
        plt.axis("off")
        plt.show()

# -----------------------------
# Save Trained Model
# -----------------------------
model.save('Efficient_classify.keras')

# -----------------------------
# Reload Model for Deployment
# -----------------------------
class_names = ['Battery', 'Keyboard', 'Microwave', 'Mobile', 'Mouse', 'PCB', 'Player', 'Printer', 'Television', 'Washing Machine']
model = tf.keras.models.load_model('Efficient_classify.keras')

# -----------------------------
# Define Inference Function for Gradio Interface
# -----------------------------
def classify_image(img):
    # Resize and preprocess input image
    img = img.resize((128, 128))
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict class
    prediction = model.predict(img_array)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence = prediction[0][index]
    
    return f"Predicted: {class_name} (Confidence: {confidence:.2f})"

# -----------------------------
# Gradio Interface for Real-Time Prediction
# -----------------------------
iface = gr.Interface(fn=classify_image, inputs=gr.Image(type="pil"), outputs="text")
iface.launch()
