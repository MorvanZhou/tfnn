import matplotlib.pyplot as plt
import numpy as np
from tfnn.evaluating.monitor import Monitor


class LineFittingMonitor(Monitor):
    def __init__(self, evaluator, figsize=(7, 7), sleep=0.0001):
        super(LineFittingMonitor, self).__init__(evaluator, 'data_fitting_monitor')
        self._sleep = sleep
        self._network = self.evaluator.network
        self._fig = plt.figure(figsize=figsize)
        self._ax = self._fig.add_subplot(111)
        self._ax.grid(True)
        self._ax.set_xlabel(r'$Real\ data$')
        self._ax.set_ylabel(r'$Predicted$')

        plt.ion()
        plt.show()

    def monitoring(self, xs, ys):
        if ys.shape[1] > 1 or xs.shape[1] > 1:
            raise NotImplementedError('Can only support ys and xs which have single value.')

        feed_dict = self.evaluator.get_feed_dict(xs, ys)
        y_predict = self._network.predictions.eval(feed_dict, self._network.sess)
        predicted_data = np.hstack((xs, y_predict))
        if not hasattr(self, '_line'):
            sorted_predicted_data = predicted_data[np.argsort(predicted_data[:, 0])]
            self._ax.scatter(xs, ys,
                             c=self.color_train,   # red like
                             s=20, alpha=0.9, label=r'$Real\ data$')
            self._line, = self._ax.plot(sorted_predicted_data[:, 0], sorted_predicted_data[:, 1],
                                        c=self.color_test,   # blue like
                                        lw=3, alpha=0.5, label=r'$Prediction$')
            self._ax.set_xlabel(r'$Input$')
            self._ax.set_ylabel(r'$Output$')
            self._ax.legend(loc='best')
        else:
            sorted_predicted_data = predicted_data[np.argsort(predicted_data[:, 0])]
            self._line.set_data(sorted_predicted_data[:, 0], sorted_predicted_data[:, 1])
            self._fig.canvas.blit(self._ax.bbox)
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()
            plt.pause(self._sleep)


