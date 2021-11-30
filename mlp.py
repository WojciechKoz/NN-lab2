import numpy as np
import activations as ac
from early_stopping import EarlyStopping
from sklearn.utils import shuffle
from utils import RANDOM_SEED
from optimizers import StandardOptimizer, MomentumOptimizer, NAGOptimizer, AdagradOptimizer, AdadeltaOptimizer
from initializers import StandardInitializer

np.random.seed(RANDOM_SEED)


class MLP:
    def __init__(self, neurons=(784, 100, 30, 10), activations=('relu', 'relu'), optimizer=StandardOptimizer(eta=0.05),
                 initializer=StandardInitializer(),
                 batch_size=100, early_stopping_max_update_interval=25, verbose=True, target_accuracy=0.97):
        if len(neurons) != len(activations) + 2:
            raise ValueError("Length of neurons has to be greater by two that length of activations")

        FUNCTIONS = {
            "relu": (ac.relu, ac.relu_prime),
            "sigmoid": (ac.sigmoid, ac.sigmoid_prime),
            "softmax": (ac.softmax, None),
            "tanh": (ac.tanh, ac.tanh_prime)
        }

        self.optimizer = optimizer
        self.activations = tuple(map(lambda label: FUNCTIONS[label][0], activations + ('softmax',)))
        self.activations_primes = tuple(map(lambda label: FUNCTIONS[label][1], activations + ('softmax',)))
        self.batch_size = batch_size
        self.verbose = verbose
        self.early_stopping = EarlyStopping(early_stopping_max_update_interval)
        self.target_accuracy = target_accuracy
        self.w, self.b = initializer.init_weights(neurons)

    def forward(self, X):
        z_vals, a_vals = [], [X.T]
        a = X.T
        for wi, bi, act in zip(self.w, self.b, self.activations):
            z = wi.dot(a) + bi
            a = act(z)
            z_vals.append(z)
            a_vals.append(a)
        return a.T, a_vals, z_vals

    def fit(self, X, y, valid_X, valid_y, epochs=25):
        losses = []
        scores = []
        for epoch in range(epochs):
            train_X, train_y = shuffle(X, y)
            X_batches = np.array_split(train_X, int(len(train_X) / self.batch_size))
            y_batches = np.array_split(train_y, int(len(train_X) / self.batch_size))
            for Xi, yi in zip(X_batches, y_batches):
                self.optimizer.update(model=self, X=Xi, y=yi)

            y_valid_pred = self.forward(valid_X)[0]
            valid_accuracy = self.accuracy(valid_y, self.to_hot_ones(y_valid_pred))
            loss = self.loss(valid_y, y_valid_pred).mean()

            scores.append(valid_accuracy)
            losses.append(loss)

            if valid_accuracy > self.target_accuracy:
                if self.verbose: print(f'Got accuracy = {self.target_accuracy}')
                break

            if self.early_stopping.check_loss(loss):
                self.early_stopping.update(loss, self.w, self.b)
            elif self.early_stopping.increase_counter():
                self.w = [wi.copy() for wi in self.early_stopping.w]
                self.b = [bi.copy() for bi in self.early_stopping.b]
                if self.verbose: print('Early Stopped')
                break

            if self.verbose:
                train_accuracy = self.accuracy(train_y, self.predict(train_X))
                self.log(epoch + 1, train_accuracy, valid_accuracy, loss, self.early_stopping.last_update_counter)

        return scores, losses

    def predict(self, X):
        y_pred, _, _ = self.forward(X)
        return self.to_hot_ones(y_pred)

    def to_hot_ones(self, y_pred):
        from utils import hot_ones
        return hot_ones(np.argmax(y_pred, axis=1), 10)

    def loss(self, y, y_pred):
        return -(np.log(y_pred) * y).sum(axis=1)

    def accuracy(self, y, y_pred):
        return np.where(y == y_pred, y, 0).sum() / len(y)

    def log(self, epoch, train_acc, test_acc, loss, last_update):
        weight_avg = sum([(w ** 2).mean() for w in self.w]) / len(self.w)
        print(f"Epoch: {epoch:3} | "
              f"train acc: {train_acc:.4f} | "
              f"valid acc: {test_acc:.4f} | "
              f"w² avg: {weight_avg:.4f} | "
              f"loss: {loss:.4f} | "
              f"last update: {last_update:2}")

