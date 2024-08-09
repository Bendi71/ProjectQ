from NeuroKit.network import NeuralNetwork
from NeuroKit.dense import Dense
from NeuroKit.activations import *
from NeuroKit.regularizer import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Generate synthetic data

def generate_multiclass_data(num_samples=1000, scale=5):
    X = np.random.randn(num_samples,2)* scale
    Y = np.zeros((num_samples, 1), dtype=int)

    # Use the first two features to determine the classes
    X_features = X[:, :2]
    Y[X_features[:, 0]**2 + X_features[:, 1]**2 < (2*scale**2)] = 0
    Y[(X_features[:, 0]**2 + X_features[:, 1]**2 >= (2*scale**2)) & (X_features[:, 0]**2 + X_features[:, 1]**2 < (4 *
                                                                                                                scale**2))] = 1
    Y[X_features[:, 0]**2 + X_features[:, 1]**2 >= (4 * scale**2)] = 2

    return X, Y

# Visualize the generated multi-class data
def plot_multiclass_data(X, Y):
    plt.scatter(X[Y.flatten() == 0][:, 0], X[Y.flatten() == 0][:, 1], color='red', label='Class 0')
    plt.scatter(X[Y.flatten() == 1][:, 0], X[Y.flatten() == 1][:, 1], color='blue', label='Class 1')
    plt.scatter(X[Y.flatten() == 2][:, 0], X[Y.flatten() == 2][:, 1], color='green', label='Class 2')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

def plot_multiclass_decision_boundary(nn, X, Y, title="Decision Boundary"):
    if Y.ndim > 1 and Y.shape[1] > 1:
        Y = np.argmax(Y, axis=1)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    grid_points = np.c_[xx.ravel(), yy.ravel()]

    grid_predictions = nn.predict(grid_points)
    grid_predictions = np.argmax(grid_predictions, axis=1)

    grid_predictions = grid_predictions.reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    cmap = ListedColormap(['red', 'blue', 'green'])
    plt.contourf(xx, yy, grid_predictions, alpha=0.4, cmap=cmap)

    unique_classes = np.unique(Y)
    colors = ['red', 'blue', 'green']
    for class_value, color in zip(unique_classes, colors):
        plt.scatter(X[Y == class_value][:, 0], X[Y == class_value][:, 1], color=color, label=f'Class {class_value}')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.show()


# Generate and plot multi-class data
X, Y = generate_multiclass_data()

plot_multiclass_data(X,Y)

Y = np.eye(3)[Y.flatten()]

nn = NeuralNetwork()
nn.add(
    Dense(2, 16, activation=Relu(), regularizer=L2(0.01)),
       Dense(16, 16, activation=Relu(), regularizer=L2(0.01)),
       Dense(16, 3, activation=Softmax())
)


nn.compile(optimizer='sgd', loss='cross_entropy', metrics=['accuracy'])
nn.earlystopping()

# train the network
nn.fit(X, Y, epochs=1000, batch_size=15)

nn.summary()

plot_multiclass_decision_boundary(nn, X, Y, title="Decision Boundary")