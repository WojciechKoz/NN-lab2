from utils import get_data
from tests import test_activations, test_neuron_number, test_eta, test_batch_size, test_weight_init_state, test_eta_decay

train_X, valid_X, train_y, valid_y, test_X, test_y = get_data()

test_neuron_number(train_X, train_y, valid_X, valid_y, test_X, test_y, verbose=False)
test_eta(train_X, train_y, valid_X, valid_y, test_X, test_y, verbose=False)
test_eta_decay(train_X, train_y, valid_X, valid_y, test_X, test_y, verbose=False)
test_batch_size(train_X, train_y, valid_X, valid_y, test_X, test_y, verbose=False)
test_weight_init_state(train_X, train_y, valid_X, valid_y, test_X, test_y, verbose=True)
test_activations(train_X, train_y, valid_X, valid_y, test_X, test_y, verbose=False)


