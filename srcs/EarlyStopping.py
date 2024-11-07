class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def check_loss(self, val_loss):
        # init best_loss
        if self.best_loss is None:
            self.best_loss = val_loss
        # if perf isn't better -> incr counter for patience
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        # else, don't stop fit
        else:
            self.best_loss = val_loss
            self.counter = 0