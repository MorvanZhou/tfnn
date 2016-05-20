import pandas as pd
import tensorflow as tf
import os, shutil


class Network(object):
    def __init__(self, n_inputs, n_outputs, input_dtype, output_dtype, output_activator,
                 do_dropout, do_l2, seed=None):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.input_dtype = input_dtype
        self.output_dtype = output_dtype
        self.output_activator = output_activator
        if do_dropout and do_l2:
            raise ValueError('Cannot do dropout and l2 at once. Choose only one of them.')
        if do_dropout:
            self.reg = 'dropout'
        if do_l2:
            self.reg = 'l2'
        if (do_dropout is False) & (do_l2 is False):
            self.reg = None
        self.seed = seed

        with tf.name_scope('Inputs'):
            self.data_placeholder = tf.placeholder(dtype=input_dtype, shape=[None, n_inputs], name='x_input')
            self.target_placeholder = tf.placeholder(dtype=output_dtype, shape=[None, n_outputs], name='y_input')
            if do_dropout:
                self.keep_prob_placeholder = tf.placeholder(dtype=tf.float32)
                tf.scalar_summary('dropout_keep_probability', self.keep_prob_placeholder)
            if do_l2:
                self.l2_placeholder = tf.placeholder(tf.float32)
                tf.scalar_summary('l2_lambda', self.l2_placeholder)
        self.layers_output = pd.Series([])
        self.layers_activated_output = pd.Series([])
        self.layers_dropped_output = pd.Series([])
        self.layers_final_output = pd.Series([])
        self.Ws = pd.Series([])
        self.bs = pd.Series([])
        self.last_layer_neurons = n_inputs
        self.last_layer_outputs = self.data_placeholder
        self.hidden_layer_number = 1
        self.has_output_layer = False

    def add_hidden_layer(self, n_neurons, activator=None):
        """
        W shape(n_last_layer_neurons, n_this_layer_neurons]
        b shape(n_this_layer_neurons, ]
        product = tf.matmul(x, W) + b
        :param n_neurons:
        :param activator:
        :return:
        """
        layer_name = 'layer_%i' % self.hidden_layer_number
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                W = self._weight_variable([self.last_layer_neurons, n_neurons])
                self._variable_summaries(W, layer_name+'/weights')
            with tf.name_scope('biases'):
                b = self._bias_variable([n_neurons, ])
                self._variable_summaries(b, layer_name + '/biases')
            with tf.name_scope('Wx_plus_b'):
                product = tf.matmul(self.last_layer_outputs, W, name='Wx') + b
            if activator is None:
                activated_product = product
            else:
                activated_product = activator(product)
            tf.histogram_summary(layer_name+'/activated_product', activated_product)
            if self.reg == 'dropout':
                dropped_product = tf.nn.dropout(activated_product,
                                                self.keep_prob_placeholder,
                                                seed=self.seed, name='dropout')
                self.layers_dropped_output.set_value(label=len(self.layers_dropped_output),
                                                     value=dropped_product)
                final_product = dropped_product
            else:
                final_product = activated_product

        self.hidden_layer_number += 1
        self.last_layer_outputs = final_product
        self.Ws.set_value(label=len(self.Ws), value=W)
        self.bs.set_value(label=len(self.bs), value=b)
        self.layers_output.set_value(label=len(self.layers_output),
                                     value=product)
        self.layers_activated_output.set_value(label=len(self.layers_output),
                                               value=activated_product)
        self.layers_final_output.set_value(label=len(self.layers_final_output),
                                           value=final_product)
        self.last_layer_neurons = n_neurons

    def add_output_layer(self, activator):
        self.add_hidden_layer(self.n_outputs, activator)
        self.has_output_layer = True

    def set_optimizer(self, optimizer, global_step=None,):
        if not self.has_output_layer:
            raise NotImplementedError('Please add output layer.')
        self._init_loss()
        with tf.name_scope('trian'):
            self.train_op = optimizer.minimize(self.loss, global_step)
        self.sess = tf.Session()
        _init = tf.initialize_all_variables()
        self.sess.run(_init)

    def run_step(self, feed_xs, feed_ys, *args):
        if self.reg == 'dropout':
            keep_prob = args[0]
            self.sess.run(self.train_op, feed_dict={self.data_placeholder: feed_xs,
                                                    self.target_placeholder: feed_ys,
                                                    self.keep_prob_placeholder: keep_prob})
        elif self.reg == 'l2':
            l2 = args[0]
            self.sess.run(self.train_op, feed_dict={self.data_placeholder: feed_xs,
                                                    self.target_placeholder: feed_ys,
                                                    self.l2_placeholder: l2})
        else:
            self.sess.run(self.train_op, feed_dict={self.data_placeholder: feed_xs,
                                                    self.target_placeholder: feed_ys})

    def get_loss(self, xs, ys):
        if self.reg == 'dropout':
            _loss_value = self.sess.run(self.loss, feed_dict={self.data_placeholder: xs,
                                                              self.target_placeholder: ys,
                                                              self.keep_prob_placeholder: 1.})
        elif self.reg == 'l2':
            _loss_value = self.sess.run(self.loss, feed_dict={self.data_placeholder: xs,
                                                              self.target_placeholder: ys,
                                                              self.l2_placeholder: 0})
        else:
            _loss_value = self.sess.run(self.loss, feed_dict={self.data_placeholder: xs,
                                                              self.target_placeholder: ys})
        return _loss_value

    def get_weights(self, layer=None):
        if not(layer is None or type(layer) is int):
            raise TypeError('layer need to be None or int')
        if layer is None:
            Ws = []
            for W_layer in self.Ws:
                W = self.sess.run(W_layer)
                Ws.append(W)
        else:
            if layer >= len(self.Ws):
                raise IndexError('Do not have layer %i' % layer)
            Ws = self.sess.run(self.Ws[layer])
        return Ws

    def _weight_variable(self, shape):
        initial = tf.random_normal(
            shape, mean=0.0, stddev=0.2, dtype=self.input_dtype, seed=self.seed, name='weights')
        return tf.Variable(initial)

    def _bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape, dtype=self.input_dtype, name='biases')
        return tf.Variable(initial)

    @staticmethod
    def _variable_summaries(var, name):
        with tf.name_scope("summaries"):
            mean = tf.reduce_mean(var)
            tf.scalar_summary('mean/' + name, mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
            tf.scalar_summary('sttdev/' + name, stddev)
            tf.scalar_summary('max/' + name, tf.reduce_max(var))
            tf.scalar_summary('min/' + name, tf.reduce_min(var))
            tf.histogram_summary(name, var)

    def _init_loss(self):
        self.loss = None
