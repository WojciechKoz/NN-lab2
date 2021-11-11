
class EarlyStopping:
    def __init__(self, max_update_wait):
        self.max_update_wait = max_update_wait
        self.last_update_counter = 0
        self.w = None
        self.b = None
        self.loss = None

    def check_loss(self, loss):
        return not self.loss or loss < self.loss

    def update(self, loss, w, b):
        self.loss = loss
        self.w = [wi.copy() for wi in w]
        self.b = [bi.copy() for bi in b]
        self.last_update_counter = 0

    def increase_counter(self):
        self.last_update_counter += 1
        return self.last_update_counter == self.max_update_wait
