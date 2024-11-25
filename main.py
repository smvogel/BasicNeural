import numpy as np
import matplotlib.pyplot as plt

# Define structure
"""
Input Layer
Hidden Layer
Output Layer
"""

def initialize_parameters(input_size, hidden_size, output_size):
    """
    Initialize weights and biases for input and hidden layers. Weights are initialized using He initialization. 
    """
    np.random.seed(42) # Seed for reproducibility
    W1 = np.random.randn(hidden_size, input_size) * np.sqrt(2 / input_size) # Weights for input layer
    b1 = np.zeros((hidden_size, 1)) # Bias for input layer
    W2 = np.random.randn(output_size, hidden_size) * np.sqrt(2 / hidden_size) # Weights for hidden layer
    b2 = np.zeros((output_size, 1)) # Bias for hidden layer
    return W1, b1, W2, b2

# Define activation function

def sigmoid(z):
    # Sigmoid activation function
   return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    # Derivative of sigmoid activation function
    return sigmoid(z) * (1 - sigmoid(z))

def relu(z):
    # ReLU activation function
    return np.maximum(0, z)

def relu_derivative(z):
    # Derivative of ReLU activation function
    return np.where(z > 0, 1, 0)

# Define forward propagation

def forward_propagation(X, W1, b1, W2, b2):
    # Input layer
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    # Hidden layer
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2) # Use sigmoid for binary classification
    return Z1, A1, Z2, A2

# Compute Loss

def compute_loss(Y, A2):
    m = Y.shape[1] # Number of samples
    loss = -1 / m * np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2))
    return loss

# Define backward propagation

def backward_propagation(X, Y, Z1, A1, Z2, A2, W1, W2):
    m= X.shape[1] # Number of samples
    
    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    
    return dW1, db1, dW2, db2

# Update parameters

def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return W1, b1, W2, b2

# Train the model

def train(X, Y, input_size, hidden_size, output_size, iterations, learning_rate):
    W1, b1, W2, b2 = initialize_parameters(input_size, hidden_size, output_size)
    
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_propagation(X, W1, b1, W2, b2)
        loss = compute_loss(Y, A2)
        dW1, db1, dW2, db2 = backward_propagation(X, Y, Z1, A1, Z2, A2, W1, W2)
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
        if i % 100 == 0:
            print(f'Iteration {i}, Loss: {loss}')
    return W1, b1, W2, b2

# Make predictions

def predict(X, W1, b1, W2, b2):
    Z1, A1, Z2, A2 = forward_propagation(X, W1, b1, W2, b2)
    predictions = np.round(A2)
    return predictions

# Test the model



X = np.array([[0, 0, 1, 1], [0, 1, 1, 0]]) # Input
Y = np.array([[0, 1, 0, 1]]) # Output
input_size = X.shape[0] # Number of input features
hidden_size = 4 # Number of neurons in hidden layer
output_size = Y.shape[0] # Number of output features
iterations = 1000 # Number of iterations
learning_rate = 0.04 # Learning rate

W1, b1, W2, b2 = train(X, Y, input_size, hidden_size, output_size, iterations, learning_rate)
predictions = predict(X, W1, b1, W2, b2)
print(f'Predictions: {predictions}')


losses = []
for i in range(iterations):
    Z1, A1, Z2, A2 = forward_propagation(X, W1, b1, W2, b2)
    loss = compute_loss(Y, A2)
    losses.append(loss)
    dW1, db1, dW2, db2 = backward_propagation(X, Y, Z1, A1, Z2, A2, W1, W2)
    W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)

plt.plot(range(iterations), losses)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss over time')
plt.show()
