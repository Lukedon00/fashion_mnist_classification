# Fashion MNIST Classification with Artificial Neural Networks (ANN)

This project demonstrates the classification of the Fashion MNIST dataset (images of clothing items) using an Artificial Neural Network (ANN) built with Keras and TensorFlow.

## Project Overview

- **Dataset**: Fashion MNIST dataset from `tensorflow.keras.datasets`
- **Algorithm**: Artificial Neural Networks (ANN)
- **Objective**: Classify clothing items (e.g., shirts, shoes, pants) based on pixel values.
- **Activation Functions**: Sigmoid, ReLU, Softmax
- **Tools Used**: Python, Keras, TensorFlow, Matplotlib, Seaborn

## Steps:
1. Load and preprocess the Fashion MNIST dataset.
2. Build a simple ANN model using Sigmoid activation.
3. Train and evaluate the model.
4. Build a deeper ANN model using Sigmoid and Softmax activation.
5. Train and evaluate the deeper model.

## Results:
- **Accuracy (Simple Model)**: Achieved ~88% accuracy with a simple ANN model.
- **Accuracy (Deeper Model)**: Achieved ~90% accuracy with a deeper ANN model.
- **Confusion Matrix**: Visualized using Seaborn heatmap.
- **Classification Report**: Includes precision, recall, and F1-score for each class.

## Why Sigmoid and ReLU?
- **Sigmoid**: Used in the simple model to classify into 10 categories. While it works, it suffers from the vanishing gradient problem in deeper networks.
- **ReLU**: Used in the deeper model's hidden layers for faster convergence and to avoid the vanishing gradient problem.
- **Softmax**: Used in the output layer for multi-class classification, which outputs probabilities for each class.

## How to Run

### Prerequisites
- Python 3.x
- Required Libraries (see `requirements.txt`)

### Steps to Run:
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/fashion_mnist_classification.git
    ```
2. Navigate to the project directory:
    ```bash
    cd fashion_mnist_classification
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the Jupyter notebook or the Python script:
    - Jupyter Notebook: Open the `fashion_mnist_classification.ipynb` file and run the cells.
    - Python Script: Execute the Python file in your terminal:
    ```bash
    python fashion_mnist_classification.py
    ```

## License
This project is licensed under the MIT License.
