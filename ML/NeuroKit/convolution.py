from .layer import Layer
import numpy as np

class Conv(Layer):
    def __init__(self, num_filters, filter_size, input_shape, stride=1, padding=0):
        super().__init__()
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.input_shape = input_shape
        self.filters = np.random.randn(num_filters, input_shape[0], filter_size, filter_size) / filter_size**2
        self.biases = np.zeros((num_filters, 1))
        self.parameters = [self.filters, self.biases]

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.convolve(input_data, self.filters, self.biases, self.stride, self.padding)
        return self.output

    def backward_propagation(self, output_gradient, optimizer):
        filters_gradient, input_gradient = self.compute_gradients(output_gradient)
        self.filters = optimizer.update(self.filters, filters_gradient)
        self.biases = optimizer.update(self.biases, np.sum(output_gradient, axis=(0, 2, 3)).reshape(self.biases.shape))
        self.parameters = [self.filters, self.biases]
        return input_gradient

    def convolve(self, input_data, filters, biases, stride, padding):
        # TODO: Implement the convolution operation
        pass

    def compute_gradients(self, output_gradient):
        # TODO: Implement the gradient computation
        pass