import pickle
from .losses import *
from .optimizer import *
from .metrics import *
from .early_stopping import *
from .activations import *
from .regularizer import *
from .dense import Dense

class NeuralNetwork(object):
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None
        self.early_stopping = None
        self.loss_functions = {
            'mse': (mse, mse_prime),
            'cross_entropy': (cross_entropy, cross_entropy_prime),
            'binary_cross_entropy': (binary_cross_entropy, binary_cross_entropy_prime)
        }
        self.metric_functions = {
            'accuracy': (accuracy),
            'R2': (r2_score)
        }

    def add(self, *layers):
        for layer in layers:
            self.layers.append(layer)

    def set_loss(self, loss_name):
        if loss_name in self.loss_functions:
            self.loss, self.loss_prime = self.loss_functions[loss_name]
        else:
            raise ValueError("Loss function not supported.")

    def set_optimizer(self, optimizer_name, **kwargs):
        if optimizer_name == 'sgd':
            self.optimizer = SGD(**kwargs)
        elif optimizer_name == 'adam':
            self.optimizer = Adam(**kwargs)
            raise ValueError("Optimizer not yet supported.")
        else:
            raise ValueError("Optimizer not supported.")

    def earlystopping(self, patience=5, min_delta=0.001):
        self.early_stopping = EarlyStopping(patience,min_delta)

    def evaluate_metrics(self, Y, Y_pred):
        results = {}
        for metric in self.metrics:
            if metric in self.metric_functions:
                results[metric] = self.metric_functions[metric](Y, Y_pred)
            else:
                raise ValueError(f"Metric {metric} not supported.")
        return results

    def predict(self, input_data):
        if input_data.ndim == 1:
            input_data = input_data[:, np.newaxis]
        output = input_data
        for layer in self.layers:
            output= layer.forward_propagation(output)

        return output

    def compile(self, optimizer,loss, metrics=None):
        self.set_loss(loss)
        self.set_optimizer(optimizer)
        if metrics:
            self.metrics = metrics

    def fit(self, X, Y, epochs, batch_size=20,verbose=True):
        if self.optimizer is None:
            raise ValueError("Optimizer not set. Use 'compile' method to set an optimizer.")
        if self.loss is None:
            raise ValueError("Loss not set. Use 'compile' method to set a loss function.")

        num_samples = X.shape[0]

        for epoch in range(epochs):
            metric_sums = {metric: 0 for metric in self.metrics}
            total_loss = 0

            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X = X[indices]
            Y = Y[indices]

            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                X_batch = X[start:end]
                Y_batch = Y[start:end]

                output = X_batch
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                loss = self.loss(Y_batch, output)
                total_loss += loss * (end - start)

                for metric in self.metrics:
                    metric_sums[metric] += self.metric_functions[metric](Y_batch, output) * (end - start)

                output_gradient = self.loss_prime(Y_batch, output)
                for layer in reversed(self.layers):
                    output_gradient = layer.backward_propagation(output_gradient, self.optimizer)

            total_loss /= num_samples
            averaged_metrics = {metric: value / num_samples for metric, value in metric_sums.items()}

            if verbose:
                metrics_str = ', '.join([f'{metric}: {value:.4f}' for metric, value in averaged_metrics.items()])
                print(f'Epoch: {epoch}, Error: {total_loss}, {metrics_str}')

            if self.early_stopping:
                if self.early_stopping.should_stop(total_loss):
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        print("Training complete.")

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
        print('Model saved.')

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)

    def summary(self):
        for i, layer in enumerate(self.layers):
            print(f'Layer {i}: {layer.__class__.__name__}')
            print(f'Input Shape: {layer.input_shape}, Output Shape: {layer.output_shape}')
            print(f'Parameters: {layer.parameters}')







