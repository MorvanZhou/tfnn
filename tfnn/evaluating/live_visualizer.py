import pyglet
import numpy as np
import copy
from matplotlib import cm


class VisualWindow(pyglet.window.Window):
    def __init__(self, network, width=800, height=600):
        super(VisualWindow, self).__init__(width, height, resizable=True,
                                           caption='Neuron Visualizer')
        self._network = network
        self.set_location(x=500, y=30)
        self.icon_image = pyglet.image.load('tfnn/evaluating/neuron.png')
        self.set_icon(self.icon_image)
        self.xs = None
        self.xs_buffer = None
        self.neurons_vertex_list = []
        x_base, y_base = self.width - 200, 100
        x0, y0, u, x_interval, y_interval = x_base, y_base, 5, 30, 4
        x1, y1 = x0 + u, y0 + u
        # (x0, y0), (x0, y1), (x1, y1), (x1, y0)
        self.neurons_batch = pyglet.graphics.Batch()
        self.layers_neuron_num = copy.deepcopy(network.record_neurons)
        self.layers_neuron_num.insert(0, network.n_inputs)
        for num_neurons in self.layers_neuron_num:
            for i in range(num_neurons):
                y0 += y_interval + u
                y1 += y_interval + u
                self.neurons_vertex_list.append(self.neurons_batch.add(4, pyglet.gl.GL_QUADS, None,
                                                                       ('v2f/static', [x0, y0, x0, y1, x1, y1, x1, y0]),
                                                                       ('c4f/stream', [0, 0, 0, 0] * 4)))
                if y0 > y_base + 200:
                    y0 = y_base
                    y1 = y0 + u
                    x0 += y_interval + u
                    x1 += y_interval + u
            x0 += x_interval
            x1 += x_interval
            y0 = y_base
            y1 = y0 + u

    def on_draw(self):
        self.clear()
        self.neurons_batch.draw()

    def update(self, dt):
        self.draw_neurons()

    def draw_neurons(self):
        if self.xs != self.xs_buffer:
            if self.xs.ndim == 1:
                self.xs = self.xs[np.newaxis, :]
            elif self.xs.shape[0] != 1:
                raise ValueError('Input data must only contain one sample')

            if self._network.reg == 'dropout':
                feed_dict = {self._network.data_placeholder: self.xs,
                             self._network.keep_prob_placeholder: self.keep_prob}
            elif self._network.reg == 'l2':
                feed_dict = {self._network.data_placeholder: self.xs,
                             self._network.l2_placeholder: self.l2_lambda}
            else:
                feed_dict = {self._network.data_placeholder: self.xs}

            layers_final_output = self._network.sess.run(
                self.network.layers_final_output, feed_dict)

            i = 0
            for neuron in range(self._network.n_inputs):
                self.neurons_vertex_list[i].colors = cm.cool(neuron[0]) * 4
                i += 1
            for layer in layers_final_output:
                for neuron in layer:
                    self.neurons_vertex_list[i].colors = cm.cool(neuron[0]) * 4
                    i += 1
            self.xs_buffer = self.xs
        else:
            pass


class LiveVisualizer(object):
    def __init__(self, network):
        self.window = VisualWindow(network, width=800, height=600)
        pyglet.clock.schedule_interval(self.window.update, 1 / 30)
        pyglet.app.run()

    def update(self, xs):
        self.window.xs = xs