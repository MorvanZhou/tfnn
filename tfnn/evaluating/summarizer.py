import os
import shutil
import tensorflow as tf


class Summarizer(object):
    def __init__(self, network, save_path='/tmp/log'):
        save_path = '/tmp/log'
        folder = save_path.split('/')[-1]
        name_len = len(folder)
        if folder in os.listdir(save_path[:-name_len]):
            shutil.rmtree(save_path)
        self.merged = tf.merge_all_summaries()
        self.train_writer = tf.train.SummaryWriter(save_path + '/train', network.sess.graph)
        self.validate_writer = tf.train.SummaryWriter(save_path + '/validate', )
        self.network = network

    def record_train(self, t_xs, t_ys, global_step, *args):
        if self.network.reg == 'dropout':
            if len(args) != 1:
                raise ValueError('Do not find keep_prob value.')
            keep_prob = args[0]
            feed_dict = {self.network.data_placeholder: t_xs,
                         self.network.target_placeholder: t_ys,
                         self.network.keep_prob_placeholder: keep_prob}
        elif self.network.reg == 'l2':
            if len(args) != 1:
                raise ValueError('Do not find l2_lambda value.')
            l2_lambda = args[0]
            feed_dict = {self.network.data_placeholder: t_xs,
                         self.network.target_placeholder: t_ys,
                         self.network.l2_placeholder: l2_lambda}
        else:
            feed_dict = {self.network.data_placeholder: t_xs,
                         self.network.target_placeholder: t_ys}
        train_result = self.network.sess.run(self.merged, feed_dict)
        self.train_writer.add_summary(train_result, global_step)

    def record_validate(self, v_xs, v_ys, global_step):
        if self.network.reg == 'dropout':
            feed_dict = {self.network.data_placeholder: v_xs,
                         self.network.target_placeholder: v_ys,
                         self.network.keep_prob_placeholder: 1.}
        elif self.network.reg == 'l2':
            feed_dict = {self.network.data_placeholder: v_xs,
                         self.network.target_placeholder: v_ys,
                         self.network.l2_placeholder: 0.}
        else:
            feed_dict = {self.network.data_placeholder: v_xs,
                         self.network.target_placeholder: v_ys}
        validate_result = self.network.sess.run(self.merged, feed_dict)
        self.validate_writer.add_summary(validate_result, global_step)


