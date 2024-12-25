import numpy as np

from layer import Layer


class Conv(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        super().__init__()
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.random(self.kernels_shape)
        self.biases = np.random.random(self.output_shape)
        self.parameters = [self.kernels, self.biases]

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.convolve(self.input, self.kernels, self.biases)
        return self.output

    def backward_propagation(self, output_gradient, optimizer):
        batch_size, input_depth, input_height, input_width = self.input.shape
        output_depth, output_height, output_width = self.output_shape

        # Initialize gradients
        kernels_gradient = np.zeros(self.kernels.shape)
        biases_gradient = np.zeros(self.biases.shape)
        input_gradient = np.zeros(self.input.shape)

        # Compute gradients
        for i in range(output_depth):
            for j in range(input_depth):
                for y in range(output_height):
                    for x in range(output_width):
                        y_start = y
                        y_end = y_start + self.kernels.shape[2]
                        x_start = x
                        x_end = x_start + self.kernels.shape[3]
                        kernels_gradient[i, j] += np.sum(
                            self.input[:, j, y_start:y_end, x_start:x_end] * output_gradient[:, i, y, x][:, np.newaxis,
                                                                             np.newaxis, np.newaxis],
                            axis=0
                        )
                        input_gradient[:, j, y_start:y_end, x_start:x_end] += np.sum(
                            self.kernels[i, j] * output_gradient[:, i, y, x][:, np.newaxis, np.newaxis, np.newaxis],
                            axis=0
                        )
            biases_gradient[i] = np.sum(output_gradient[:, i, :, :], axis=(0, 1, 2))

        # Update parameters
        self.kernels = optimizer.update(self.kernels, kernels_gradient)
        self.biases = optimizer.update(self.biases, biases_gradient)

        return input_gradient

    def convolve(self, input_data, kernels, biases):
        batch_size, input_depth, input_height, input_width = input_data.shape
        output_depth, output_height, output_width = self.output_shape

        # Initialize the output with biases
        output = np.zeros((batch_size, output_depth, output_height, output_width))

        # Perform the convolution operation
        for i in range(output_depth):
            for j in range(input_depth):
                for y in range(output_height):
                    for x in range(output_width):
                        y_start = y
                        y_end = y_start + kernels.shape[2]
                        x_start = x
                        x_end = x_start + kernels.shape[3]
                        output[:, i, y, x] += np.sum(input_data[:, j, y_start:y_end, x_start:x_end] * kernels[i, j],
                                                     axis=(1, 2))
            output[:, i, :, :] += biases[i]

        return output


def test_backward_propagation():
    # Initialize the Conv layer
    input_shape = (3, 32, 32)  # Example input shape (depth, height, width)
    kernel_size = 3
    depth = 8
    conv_layer = Conv(input_shape, kernel_size, depth)

    # Create a sample input data array with batch size of 1
    input_data = np.random.random((1, *input_shape))

    # Perform forward propagation
    output = conv_layer.forward_propagation(input_data)

    # Create a sample output gradient array
    output_gradient = np.random.random(output.shape)

    # Define a simple optimizer with an update method
    class SimpleOptimizer:
        def update(self, param, grad):
            learning_rate = 0.01
            return param - learning_rate * grad

    optimizer = SimpleOptimizer()

    # Perform backward propagation
    input_gradient = conv_layer.backward_propagation(output_gradient, optimizer)

    # Print the gradients and updated parameters
    print("Kernels gradient:", conv_layer.kernels)
    print("Biases gradient:", conv_layer.biases)
    print("Input gradient:", input_gradient)


# Run the test
test_backward_propagation()
