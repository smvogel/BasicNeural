Basic XOR problem solving using a Neural Network

1. Imports

import numpy as np
import matplotlib.pyplot as plt
	•	numpy: Used for numerical operations such as matrix multiplication and random number generation.
	•	matplotlib.pyplot: Used for plotting the loss over time.

2. Program Structure and Layers

	•	Input Layer: Receives input features (X) of size input_size.
	•	Hidden Layer: Contains hidden_size neurons and introduces non-linearity via activation functions (ReLU in this case).
	•	Output Layer: Produces predictions (A2), one per output neuron. Sigmoid activation is used for binary classification.

3. Initializing Parameters

	•	Weights (W1, W2): Randomly initialized using He initialization to prevent vanishing/exploding gradients.
	•	Dimensions:
	•	W1: (hidden_size, input_size)
	•	W2: (output_size, hidden_size)

	•	Biases (b1, b2):
	•	Initialized to zeros.
	•	Dimensions:
	•	b1: (hidden_size, 1)
	•	b2: (output_size, 1)

4. Activation Functions

	•	Sigmoid: Used in the output layer for binary classification. Maps values to [0, 1].
	•	ReLU: Used in the hidden layer. Maps negative values to 0, retaining positive values.
	•	Derivatives: Used in backpropagation to compute gradients for weight updates.

5. Forward Propagation

	•	Input (X): Multiplied with weights and added to biases.
	•	Activations:
	•	Hidden layer: ReLU activation.
	•	Output layer: Sigmoid activation for binary classification.
	•	Outputs:
	•	Z1, Z2: Pre-activation values.
	•	A1, A2: Post-activation values.

6. Loss Function

	•	Cross-Entropy Loss: Measures the error between predicted probabilities (A2) and true labels (Y).

7. Backward Propagation

	•	Computes gradients (dW1, db1, dW2, db2) using the chain rule to adjust weights and biases.
	•	relu_derivative and sigmoid_derivative are used to propagate errors backward through activations.

8. Updating Parameters

	•	Weights and biases are updated using gradient descent.

9. Training the Model

	•	Trains the neural network for iterations epochs, printing the loss every 100 iterations.

10. Making Predictions

	•	Uses forward propagation to make predictions (A2) and rounds them to binary outputs (0 or 1).

11. Testing and Plotting

    X = np.array([[0, 0, 1, 1], [0, 1, 1, 0]]) # Input
    Y = np.array([[0, 1, 0, 1]]) # Output

	•	XOR problem: Inputs X and labels Y.
	•	Visualizes the loss reduction over training epochs.
