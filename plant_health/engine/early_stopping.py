class EarlyStopping:
    """Stops training when the validation indicator (F1 by default) has not improved
    for ``patience`` evaluations (Sec. 3.2: "early stopping was implemented")."""

    def __init__(self, patience=10, mode='max', min_delta=0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = None
        self.counter = 0

    def step(self, value):
        """Returns (improved, should_stop)."""
        if self.best is None or self._better(value):
            self.best = value
            self.counter = 0
            return True, False
        self.counter += 1
        return False, self.patience > 0 and self.counter >= self.patience

    def _better(self, value):
        if self.mode == 'max':
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta
