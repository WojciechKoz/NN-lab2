from utils import get_data
from tests import test_optimizer, test_initializer
from optimizers import StandardOptimizer, MomentumOptimizer, NAGOptimizer, AdagradOptimizer, AdadeltaOptimizer, AdamOptimizer
from initializers import StandardInitializer, XavierInitializer, HeInitializer

train_X, valid_X, train_y, valid_y, test_X, test_y = get_data()
opts = [StandardOptimizer, MomentumOptimizer, NAGOptimizer, AdagradOptimizer, AdadeltaOptimizer, AdamOptimizer]
opts = [AdamOptimizer]
opt_label = ["Standard", "Momentum", "Nastrov", "Adagrad", "Adadelta", "Adam"]
opt_label = ["Adam"]

inits = [StandardInitializer, XavierInitializer, HeInitializer]
init_labels = ["Standard", "Xavier", "He"]
init = inits[2]()

'''
print("Optimizers")
for opt, label in zip(opts, opt_label):
    print(label)
    test_optimizer(train_X, train_y, valid_X, valid_y, test_X, test_y, opt=opt(), verbose=True)
'''

print("Inits")
for init, label in zip(inits, init_labels):
    print(label)
    test_initializer(train_X, train_y, valid_X, valid_y, test_X, test_y, init=init(), verbose=False)
