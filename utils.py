import numpy as np
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from time import time
from initializers import StandardInitializer
from optimizers import StandardOptimizer

RANDOM_SEED = 0

def flatten(X):
    return X.reshape(X.shape[0], X.shape[1] * X.shape[2])


def hot_ones(y, fixed_size=10):
    output = np.zeros((y.size, fixed_size))
    output[np.arange(y.size), y] = 1
    return output


def get_data(valid_size=0.05):
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_X = flatten(train_X / 255.0)
    test_X = flatten(test_X / 255.0)
    train_y = hot_ones(train_y, 10)
    test_y = hot_ones(test_y, 10)
    train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=valid_size,
                                                          random_state=RANDOM_SEED)
    return train_X, valid_X, train_y, valid_y, test_X, test_y


def plot_accuracy_and_loss(accuracy, losses):
    plt.plot(range(len(accuracy)), accuracy, label='Accuracy')
    plt.plot(range(len(losses)), losses, label='Loss')
    plt.legend()
    plt.title('Loss and accuracy')
    plt.xlabel('Epochs')
    plt.hlines([0, 1], 0, len(accuracy), linestyles='dotted', colors='black')
    plt.show()


def test_model(train_X, train_y, valid_X, valid_y, test_X, test_y, opt=StandardOptimizer(),
               eta=0.05, batch_size=100, neurons=(784, 100, 30, 10),
               activations=('sigmoid', 'sigmoid'), init=StandardInitializer(), experiments_n=10, verbose=False):
    from mlp import MLP
    epochs_sum = 0
    accuracy_sum = 0
    time_sum = 0
    accuracies = [[] for _ in range(25)]

    for _ in range(experiments_n):
        model = MLP(batch_size=batch_size, neurons=neurons, optimizer=opt,
                    activations=activations, initializer=init, verbose=verbose)

        start = time()
        accuracy, losses = model.fit(train_X.copy(), train_y.copy(), valid_X.copy(), valid_y.copy())
        time_sum += time() - start

        for i in range(min(len(accuracies), len(accuracy))):
            accuracies[i].append(accuracy[i])

        if verbose:
            plot_accuracy_and_loss(accuracy, losses)
        epochs_sum += len(accuracy)
        accuracy_sum += model.accuracy(test_y, model.predict(test_X))

    for i in range(len(accuracies)):
        if accuracies[i]:
            accuracies[i] = sum(accuracies[i])/len(accuracies[i])
        else:
            accuracies[i] = None
    accuracies = list(filter(lambda x: x is not None, accuracies))

    return accuracy_sum / experiments_n, epochs_sum / experiments_n, time_sum / experiments_n, accuracies
