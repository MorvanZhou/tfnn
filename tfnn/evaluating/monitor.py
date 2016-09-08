import matplotlib.pyplot as plt


class Monitor(object):
    def __init__(self, evaluator, name):
        self.evaluator = evaluator
        self.name = name

    def monitoring(self, *args, **kwargs):
        pass

    @staticmethod
    def hold_plot():
        plt.ioff()
        plt.show()

