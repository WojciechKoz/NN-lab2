import numpy as np


def forward(w, b, X, model):
    z_vals, a_vals = [], [X.T]
    a = X.T
    for wi, bi, act in zip(w, b, model.activations):
        z = wi.dot(a) + bi
        a = act(z)
        z_vals.append(z)
        a_vals.append(a)
    return a.T, a_vals, z_vals


def backward(y_pred, y, a, z, w, b, model):
    deltas = [(y - y_pred).T]

    for i in range(len(w) - 2, -1, -1):
        deltas.append(w[i + 1].T.dot(deltas[-1]) * model.activations_primes[i](z[i]))

    deltas.reverse()
    w_grad = []
    b_grad = []
    for i in range(len(w)):
        w_grad.append(model.optimizer.optimize(i, "w", deltas[i].dot(a[i].T)/model.batch_size))
        b_grad.append(model.optimizer.optimize(i, "b", np.array([deltas[i].sum(axis=1)]).T / model.batch_size))
    return w_grad, b_grad


class StandardOptimizer:
    def __init__(self, eta=0.05):
        self.eta = eta

    def update(self, model, X, y):
        y_pred, a, z = forward(model.w, model.b, X, model)
        w_grad, b_grad = backward(y_pred, y, a, z, model.w, model.b, model)

        for i, (dw, db) in enumerate(zip(w_grad, b_grad)):
            model.w[i] += dw
            model.b[i] += db

    def optimize(self, layer_nr, type_, gradient):
        return self.eta * gradient


class MomentumOptimizer:
    def __init__(self, eta=0.05, gamma=0.8):
        self.previous_grads = []
        self.gamma = gamma
        self.eta = eta

    def update(self, model, X, y):
        y_pred, a, z = forward(model.w, model.b, X, model)
        w_grad, b_grad = backward(y_pred, y, a, z, model.w, model.b, model)

        for i, (dw, db) in enumerate(zip(w_grad, b_grad)):
            model.w[i] += dw
            model.b[i] += db

    def optimize(self, layer_nr, type, gradient):
        if layer_nr == len(self.previous_grads):
            self.previous_grads.append({type: np.zeros_like(gradient)})
        elif type not in self.previous_grads[layer_nr].keys():
            self.previous_grads[layer_nr][type] = np.zeros_like(gradient)

        output = self.gamma * self.previous_grads[layer_nr][type] + gradient
        self.previous_grads[layer_nr][type] = output
        return self.eta * output


class NAGOptimizer:
    def __init__(self, eta=0.05, gamma=0.8):
        self.gamma = gamma
        self.previous_grads = []
        self.eta = eta

    def update(self, model, X, y):
        if not self.previous_grads:
            self.init_prev_grads(model)

        w = [wi + self.gamma * pi["w"] for wi, pi in zip(model.w, self.previous_grads)]
        b = [bi + self.gamma * pi["b"] for bi, pi in zip(model.b, self.previous_grads)]

        y_pred, a, z = forward(w, b, X, model)
        w_grad, b_grad = backward(y_pred, y, a, z, w, b, model)

        for i, (dw, db) in enumerate(zip(w_grad, b_grad)):
            model.w[i] += dw
            model.b[i] += db

    def init_prev_grads(self, model):
        for w, b in zip(model.w, model.b):
            self.previous_grads.append({"w": np.zeros_like(w), "b": np.zeros_like(b)})

    def optimize(self, layer_nr, type_, gradient):
        output = self.gamma * self.previous_grads[layer_nr][type_] + self.eta * gradient
        self.previous_grads[layer_nr][type_] = output
        return output


class AdagradOptimizer:
    def __init__(self, eta=0.05, epsilon=1e-2):
        self.epsilon = epsilon
        self.sum_of_prev_grads = []
        self.eta = eta

    def init_G(self, model):
        for w, b in zip(model.w, model.b):
            self.sum_of_prev_grads.append({"w": np.zeros_like(np.square(w)), "b": np.zeros_like(np.square(b))})

    def update(self, model, X, y):
        if not self.sum_of_prev_grads:
            self.init_G(model)
        y_pred, a, z = forward(model.w, model.b, X, model)
        w_grad, b_grad = backward(y_pred, y, a, z, model.w, model.b, model)

        for i, (dw, db) in enumerate(zip(w_grad, b_grad)):
            model.w[i] += dw
            model.b[i] += db

    def optimize(self, layer_nr, type_, gradient):
        self.sum_of_prev_grads[layer_nr][type_] += np.square(gradient)
        multiplier = 1 / np.sqrt(self.sum_of_prev_grads[layer_nr][type_] + self.epsilon)
        output = multiplier * gradient
        return self.eta * output


class AdadeltaOptimizer:
    def __init__(self, eta=0.03, gamma=0.9, epsilon=1e-3):
        self.epsilon = epsilon
        self.gamma = gamma
        self.prev_E_g = []
        self.prev_E_w = []
        self.eta = eta

    def init_prev_Es(self, model):
        for w, b in zip(model.w, model.b):
            self.prev_E_g.append({"w": np.zeros_like(w), "b": np.zeros_like(b)})
            self.prev_E_w.append({"w": np.zeros_like(w), "b": np.zeros_like(b)})

    def update(self, model, X, y):
        if not self.prev_E_w:
            self.init_prev_Es(model)
        y_pred, a, z = forward(model.w, model.b, X, model)
        w_grad, b_grad = backward(y_pred, y, a, z, model.w, model.b, model)

        for i, (dw, db) in enumerate(zip(w_grad, b_grad)):
            model.w[i] += dw
            model.b[i] += db

    def optimize(self, layer_nr, type_, gradient):
        E_g = self.gamma * self.prev_E_g[layer_nr][type_].mean() + (1. - self.gamma) * np.square(gradient)

        delta_w = gradient / np.sqrt(E_g.mean() + self.epsilon)
        E_w = self.gamma * self.prev_E_w[layer_nr][type_].mean() + (1 - self.gamma) * np.square(delta_w)

        RMS_w = np.sqrt(E_w)
        RMS_g = np.sqrt(E_g + self.epsilon)
        output = RMS_w / RMS_g * gradient

        self.prev_E_g[layer_nr][type_] = E_g
        self.prev_E_w[layer_nr][type_] = E_w

        return self.eta * output


class AdamOptimizer:
    def __init__(self, eta=0.005, beta1=0.9, beta2=0.999, epsilon=1e-2):
        self.epsilon = epsilon
        self.t = 1
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = []
        self.v = []
        self.eta = eta

    def init_params(self, model):
        for w, b in zip(model.w, model.b):
            self.m.append({"w": np.zeros_like(w), "b": np.zeros_like(b)})
            self.v.append({"w": np.zeros_like(w), "b": np.zeros_like(b)})

    def update(self, model, X, y):
        if not self.m:
            self.init_params(model)

        y_pred, a, z = forward(model.w, model.b, X, model)
        w_grad, b_grad = backward(y_pred, y, a, z, model.w, model.b, model)

        for i, (dw, db) in enumerate(zip(w_grad, b_grad)):
            model.w[i] += dw
            model.b[i] += db

    def optimize(self, layer_nr, type_, gradient):
        self.m[layer_nr][type_] = self.beta1 * self.m[layer_nr][type_] + (1 - self.beta1) * gradient
        self.v[layer_nr][type_] = self.beta2 * self.v[layer_nr][type_] + (1 - self.beta2) * (gradient ** 2)

        m_corr = self.m[layer_nr][type_] / (1 - self.beta1 ** self.t)
        v_corr = self.v[layer_nr][type_] / (1 - self.beta2 ** self.t)

        self.t += 1

        return self.eta * m_corr / (np.sqrt(v_corr) + self.epsilon)
