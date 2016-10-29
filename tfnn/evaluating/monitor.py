import matplotlib.pyplot as plt


class Monitor(object):
    def __init__(self, evaluator, name):
        self.evaluator = evaluator
        self.name = name
        self.color_train = '#F94A25'    # red like
        self.color_test = '#0690EF'     # blue like

    def monitoring(self, *args, **kwargs):
        pass


