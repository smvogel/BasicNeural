import java.util.Random;
import java.util.Arrays;
/*
 * This program implements a simple neural network with one hidden layer to solve the XOR problem. A binary classification problem.
 * Neural network has 2 input units, 4 hidden units, and 1 output unit. The activation function for the hidden layer is ReLU and for the output layer is sigmoid.
 * The loss function is binary cross-entropy, and weights are initialized using He initialization. 
 * The neural network is trained using mini-batch gradient descent. Lastly, program prints the loss every 100 iterations and makes predictions on the XOR inputs.
 */


public class NeuralNetwork {
    private double[][] W1, W2;  // Weights
    private double[] b1, b2;   // Biases
    private int inputSize, hiddenSize, outputSize;

    // Constructor to initialize parameters
    public NeuralNetwork(int inputSize, int hiddenSize, int outputSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;

        // Initialize weights and biases
        this.W1 = initializeWeights(hiddenSize, inputSize);
        this.b1 = new double[hiddenSize];
        this.W2 = initializeWeights(outputSize, hiddenSize);
        this.b2 = new double[outputSize];
    }

    // Initialize weights using He initialization
    private double[][] initializeWeights(int rows, int cols) {
        Random random = new Random();
        double[][] weights = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                weights[i][j] = random.nextGaussian() * Math.sqrt(2.0 / cols);
            }
        }
        return weights;
    }

    private double[] sigmoid(double[] z) {
        double[] result = new double[z.length];
        for (int i = 0; i < z.length; i++) {
            result[i] = 1 / (1 + Math.exp(-z[i]));
        }
        return result;
    }

    private double[] relu(double[] z) {
        double[] result = new double[z.length];
        for (int i = 0; i < z.length; i++) {
            result[i] = Math.max(0, z[i]);
        }
        return result;
    }

    private double[] reluDerivative(double[] z) {
        double[] result = new double[z.length];
        for (int i = 0; i < z.length; i++) {
            result[i] = z[i] > 0 ? 1 : 0;
        }
        return result;
    }

    // Forward propagation
    private double[] forward(double[] input, double[][] weights, double[] bias) {
        double[] z = new double[weights.length];
        for (int i = 0; i < weights.length; i++) {
            z[i] = bias[i];
            for (int j = 0; j < input.length; j++) {
                z[i] += weights[i][j] * input[j];
            }
        }
        return z;
    }

    // Compute loss (binary cross-entropy)
    private double computeLoss(double[] Y, double[] A) {
        double loss = 0;
        int m = Y.length;
        for (int i = 0; i < m; i++) {
            loss += -Y[i] * Math.log(A[i]) - (1 - Y[i]) * Math.log(1 - A[i]);
        }
        return loss / m;
    }

    // Train the neural network
    public void train(double[][] X, double[][] Y, int iterations, double learningRate) {
        int m = X.length; // Number of training examples
        for (int iter = 0; iter < iterations; iter++) {
            double[][] A1 = new double[m][hiddenSize];
            double[][] Z1 = new double[m][hiddenSize];
            double[][] A2 = new double[m][outputSize];
            double[][] Z2 = new double[m][outputSize];

            // Forward propagation
            for (int i = 0; i < m; i++) {
                Z1[i] = forward(X[i], W1, b1);
                A1[i] = relu(Z1[i]);
                Z2[i] = forward(A1[i], W2, b2);
                A2[i] = sigmoid(Z2[i]);
            }

            // Compute loss
            double loss = 0;
            for (int i = 0; i < m; i++) {
                loss += computeLoss(Y[i], A2[i]);
            }
            loss /= m;

            // Print every 100 iterations
            if (iter % 100 == 0) {
                System.out.println("Iteration " + iter + ", Loss: " + loss);
            }

            // Backpropagation
            double[][] dW2 = new double[outputSize][hiddenSize];
            double[] db2 = new double[outputSize];
            double[][] dW1 = new double[hiddenSize][inputSize];
            double[] db1 = new double[hiddenSize];

            for (int i = 0; i < m; i++) {
                double[] dZ2 = new double[outputSize];
                for (int j = 0; j < outputSize; j++) {
                    dZ2[j] = A2[i][j] - Y[i][j];
                    db2[j] += dZ2[j];
                    for (int k = 0; k < hiddenSize; k++) {
                        dW2[j][k] += dZ2[j] * A1[i][k];
                    }
                }

                double[] dA1 = new double[hiddenSize];
                for (int j = 0; j < hiddenSize; j++) {
                    for (int k = 0; k < outputSize; k++) {
                        dA1[j] += dZ2[k] * W2[k][j];
                    }
                }

                double[] dZ1 = reluDerivative(Z1[i]);
                for (int j = 0; j < hiddenSize; j++) {
                    dZ1[j] *= dA1[j];
                    db1[j] += dZ1[j];
                    for (int k = 0; k < inputSize; k++) {
                        dW1[j][k] += dZ1[j] * X[i][k];
                    }
                }
            }

            // Update weights and biases
            for (int i = 0; i < outputSize; i++) {
                db2[i] /= m;
                for (int j = 0; j < hiddenSize; j++) {
                    dW2[i][j] /= m;
                    W2[i][j] -= learningRate * dW2[i][j];
                }
                b2[i] -= learningRate * db2[i];
            }

            for (int i = 0; i < hiddenSize; i++) {
                db1[i] /= m;
                for (int j = 0; j < inputSize; j++) {
                    dW1[i][j] /= m;
                    W1[i][j] -= learningRate * dW1[i][j];
                }
                b1[i] -= learningRate * db1[i];
            }
        }
    }

    // Make predictions
    public double[] predict(double[] input) {
        double[] Z1 = forward(input, W1, b1);
        double[] A1 = relu(Z1);
        double[] Z2 = forward(A1, W2, b2);
        double[] A2 = sigmoid(Z2);
        return A2;
    }

    public static void main(String[] args) {
        // XOR inputs and outputs
        double[][] X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        double[][] Y = {{0}, {1}, {1}, {0}};

        // Create neural network
        NeuralNetwork nn = new NeuralNetwork(2, 4, 1);

        // Train neural network
        nn.train(X, Y, 5000, 0.1);

        // Test predictions
        for (double[] input : X) {
            double[] output = nn.predict(input);
            System.out.println("Input: " + Arrays.toString(input) + ", Prediction: " + Arrays.toString(output));
            System.out.println("Rounded Prediction: " + Math.round(output[0]));
        }
    }
}