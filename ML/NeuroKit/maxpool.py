from .layer import Layer
import numpy as np

class MaxPooling(Layer):
    # TODO: revise the class
    def __init__(self, pool_size, stride):
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.pool(input_data, self.pool_size, self.stride)
        return self.output

    def backward_propagation(self, output_gradient):
        input_gradient = self.compute_gradients(output_gradient)
        return input_gradient

    def pool(self, input_data, pool_size, stride):
        # Implement the max pooling operation
        batch_size, depth, height, width = input_data.shape
        pooled_height = (height - pool_size) // stride + 1
        pooled_width = (width - pool_size) // stride + 1
        pooled = np.zeros((batch_size, depth, pooled_height, pooled_width))

        for i in range(pooled_height):
            for j in range(pooled_width):
                h_start = i * stride
                h_end = h_start + pool_size
                w_start = j * stride
                w_end = w_start + pool_size
                pooled[:, :, i, j] = np.max(input_data[:, :, h_start:h_end, w_start:w_end], axis=(2, 3))

        return pooled

    def compute_gradients(self, output_gradient):
        # Implement the gradient computation for max pooling
        batch_size, depth, height, width = self.input.shape
        input_gradient = np.zeros_like(self.input)
        pooled_height = (height - self.pool_size) // self.stride + 1
        pooled_width = (width - self.pool_size) // self.stride + 1

        for i in range(pooled_height):
            for j in range(pooled_width):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size

                input_slice = self.input[:, :, h_start:h_end, w_start:w_end]
                max_pool = np.max(input_slice, axis=(2, 3), keepdims=True)
                mask = (input_slice == max_pool)
                input_gradient[:, :, h_start:h_end, w_start:w_end] += mask * output_gradient[:, :, i, j][:, :, None, None]

        return input_gradient