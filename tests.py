from utils import test_model


def test_neuron_number(train_X, train_y, valid_X, valid_y, test_X, test_y, verbose=True):
    results = []
    neurons = [(784, 25, 10), (784, 50, 10), (784, 100, 10), (784, 200, 10), (784, 100, 50, 10)]
    activations = [('relu',), ('relu',), ('relu',), ('relu',), ('relu', 'relu')]

    for n, act in zip(neurons, activations):
        print(f"===============  TESTING: neurons = {n} =================")
        results.append(test_model(train_X, train_y, valid_X, valid_y, test_X, test_y, neurons=n, activations=act,
                                  verbose=verbose))

    for n, (accuracy, epochs, time) in zip(neurons, results):
        print(f"neurons={n} | accuracy={accuracy} | epochs={epochs} | avg time={time}")


def test_eta(train_X, train_y, valid_X, valid_y, test_X, test_y, verbose=True):
    etas = [0.005, 0.01, 0.05, 0.1]
    results = []
    for eta in etas:
        print(f"===============  TESTING: eta = {eta} =================")
        results.append(test_model(train_X, train_y, valid_X, valid_y, test_X, test_y, eta=eta, eta_decay=1, verbose=verbose))
    for eta, (accuracy, epochs, time) in zip(etas, results):
        print(f"eta={eta} | accuracy={accuracy} | epochs={epochs} | time={time}")

def test_eta_decay(train_X, train_y, valid_X, valid_y, test_X, test_y, verbose=True):
    ETA = 0.05
    eta_decays = [0.9, 0.95, 0.99, 0.999, 1]
    results = []
    for decay in eta_decays:
        print(f"===============  TESTING: eta decay = {decay} =================")
        results.append(test_model(train_X, train_y, valid_X, valid_y, test_X, test_y, eta=ETA, eta_decay=decay, verbose=verbose))
    for decay, (accuracy, epochs, time) in zip(eta_decays, results):
        print(f"eta decay={decay} | accuracy={accuracy} | epochs={epochs} | time={time}")



def test_batch_size(train_X, train_y, valid_X, valid_y, test_X, test_y, verbose=True):
    batch_sizes = [10, 25, 50, 100, 250]
    results = []
    for batch_size in batch_sizes:
        print(f"===============  TESTING: batch = {batch_size} =================")
        results.append(test_model(train_X, train_y, valid_X, valid_y, test_X, test_y, batch_size=batch_size,
                                  verbose=verbose))
    for batch_size, (accuracy, epochs, time) in zip(batch_sizes, results):
        print(f"batch size={batch_size} | accuracy={accuracy} | epochs={epochs} | time={time}")


def test_weight_init_state(train_X, train_y, valid_X, valid_y, test_X, test_y, verbose=True):
    states = [(0, 0.01), (0, 0.05), (0, 0.1), (0, 0.5), (0, 1)]
    results = []
    for w_mean, w_sigma in states:
        print(f"===============  TESTING: N({w_mean}, {w_sigma}) =================")
        results.append(test_model(train_X, train_y, valid_X, valid_y, test_X, test_y, w_mean=w_mean, w_sigma=w_sigma,
                                  verbose=verbose))
    for (w_mean, w_sigma), (accuracy, epochs, time) in zip(states, results):
        print(f"weights ~ N({w_mean}, {w_sigma}) | accuracy={accuracy} | epochs={epochs} | time={time}")


def test_activations(train_X, train_y, valid_X, valid_y, test_X, test_y, verbose=True):
    results = []
    neurons = (784, 100, 30, 10)
    activations = [('relu','relu'), ('sigmoid','sigmoid'), ('tanh','tanh')]

    for act in activations:
        print(f"===============  TESTING: {act[0]} =================")
        results.append(test_model(train_X, train_y, valid_X, valid_y, test_X, test_y, neurons=neurons,
                                  activations=act, verbose=verbose))

    for act, (accuracy, epochs, time) in zip(activations, results):
        print(f"activation={act[0]} | accuracy={accuracy} | epochs={epochs} | time={time}")
