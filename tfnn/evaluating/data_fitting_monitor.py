import matplotlib.pyplot as plt
import numpy as np
from tfnn.evaluating.monitor import Monitor


class DataFittingMonitor(Monitor):
    def __init__(self, evaluator, figsize=(7, 7), sleep=0.0001):
        super(DataFittingMonitor, self).__init__(evaluator, 'data_fitting_monitor')
        self._sleep = sleep
        self._network = self.evaluator.network
        self._fig = plt.figure(figsize=figsize)
        self._ax = self._fig.add_subplot(111)

        self._scat = self._ax.scatter(
            [], [],
            label=r'$predicted$',
            c=self.color_test,      # blue like
            s=30, alpha=0.6, edgecolor='none')
        self._real_line, = self._ax.plot(
            [None, None], [None, None],
            ls='--', lw=3,
            c=self.color_train,        # red like
            label=r'$real$')
        self._ax.grid(True)
        self._ax.legend(loc='upper left')
        self._ax.set_xlabel(r'$Real\ data$')
        self._ax.set_ylabel(r'$Predicted$')

        plt.ion()
        plt.show()

    def monitoring(self, xs, ys):
        if ys.shape[1] > 1:
            raise NotImplementedError('Can only support ys which have single value.')

        feed_dict = self.evaluator.get_feed_dict(xs, ys)
        y_predict = self._network.predictions.eval(feed_dict, self._network.sess)
        # scatter change data.
        self._scat.set_offsets(np.hstack((ys, y_predict)))
        # scatter change color:
        # self._scat_axes.set_array(...)
        y_real_max, y_real_min = ys.min(), ys.max()
        self._real_line.set_data([y_real_min, y_real_max], [y_real_min, y_real_max])
        offset = 0.1 * (y_real_max - y_real_min)
        self._ax.set_ylim([y_real_min - offset, y_real_max + offset])
        self._ax.set_xlim([y_real_min - offset, y_real_max + offset])
        self._fig.canvas.blit(self._ax.bbox)
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        plt.pause(self._sleep)


