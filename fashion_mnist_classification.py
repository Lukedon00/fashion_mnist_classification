# Fashion MNIST Classification using Artificial Neural Networks (ANN)

# Import necessary libraries
import tensorflow as tf # type: ignore
from tensorflow import keras # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns # type: ignore

# Load the Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Normalize the data (scaling pixel values to the range 0-1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Flatten the data (28x28 images to 784-dimensional vectors)
x_train_flattened = x_train.reshape(len(x_train), 28*28)
x_test_flattened = x_test.reshape(len(x_test), 28*28)

# Build a simple ANN model with sigmoid activation
model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')  # Output layer with 10 neurons (for 10 classes)
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model for 5 epochs
model.fit(x_train_flattened, y_train, epochs=5)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(x_test_flattened, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Make predictions on the test set
y_predicted = model.predict(x_test_flattened)

# Convert predicted probabilities to class labels
y_predicted_labels = [np.argmax(i) for i in y_predicted]

# Display the first few predictions alongside actual labels
print(f"Predicted labels: {y_predicted_labels[:12]}")
print(f"Actual labels: {y_test[:12]}")

# Confusion Matrix
cm = tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_labels)

# Plot the confusion matrix using a heatmap
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Print classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_predicted_labels))

# Build a deeper ANN model with multiple layers and sigmoid activation
model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation='sigmoid'),  # First hidden layer
    keras.layers.Dense(200, activation='sigmoid'),                      # Second hidden layer
    keras.layers.Dense(10, activation='softmax')                        # Output layer with softmax activation
])

# Compile the deeper model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the deeper model for 10 epochs
model.fit(x_train_flattened, y_train, epochs=10)

# Evaluate the deeper model
test_loss, test_accuracy = model.evaluate(x_test_flattened, y_test)
print(f"Test Accuracy (Deeper Model): {test_accuracy * 100:.2f}%")

# Make predictions with the deeper model
y_predicted = model.predict(x_test_flattened)
y_predicted_labels = [np.argmax(i) for i in y_predicted]

# Confusion matrix for deeper model
cm = tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_labels)

# Plot confusion matrix for deeper model
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Print classification report for deeper model
print(classification_report(y_test, y_predicted_labels))
