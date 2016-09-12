import os
import shutil
import tfnn


class Summarizer(object):
    def __init__(self, network=None, save_path='/tmp',):
        if network is not None:
            self._network = network
            check_dir = os.getcwd() + save_path
            if os.path.isdir(check_dir):
                self.save_path = check_dir
            elif os.path.isdir(save_path):
                self.save_path = save_path
            else:
                raise NotADirectoryError('the directory is not exist: %s' % save_path)

            self._folder = 'tensorflow_logs'
            if self._folder in os.listdir(self.save_path):
                shutil.rmtree(self.save_path+'/' + self._folder)
            self.merged = tfnn.merge_all_summaries()

    def record_train(self, t_xs, t_ys,):
        if not hasattr(self, 'train_writer'):
            self.train_writer = tfnn.train.SummaryWriter(self.save_path + '/' + self._folder + '/train',
                                                         self._network.sess.graph)
        if self._network.reg == 'dropout':
            value_pass_in = self._network.sess.run(self._network.keep_prob)
        elif self._network.reg == 'l2':
            value_pass_in = self._network.sess.run(self._network.l2_value)
        else:
            value_pass_in = None
        global_step = self._network.sess.run(self._network.global_step)
        feed_dict = self._get_feed_dict(t_xs, t_ys, value_pass_in)
        train_result = self._network.sess.run(self.merged, feed_dict)
        self.train_writer.add_summary(train_result, global_step)

    def record_test(self, v_xs, v_ys):
        if not hasattr(self, 'test_writer'):
            self.test_writer = tfnn.train.SummaryWriter(self.save_path + '/' + self._folder + '/test', )
        if self._network.reg == 'dropout':
            value_pass_in = 1.
        elif self._network.reg == 'l2':
            value_pass_in = 0.
        else:
            value_pass_in = None
        global_step = self._network.sess.run(self._network.global_step)
        feed_dict = self._get_feed_dict(v_xs, v_ys, value_pass_in)
        test_result = self._network.sess.run(self.merged, feed_dict)
        self.test_writer.add_summary(test_result, global_step)

    def web_visualize(self, path=None):
        if (path is None) and (self._network is not None):
            whole_path = self.save_path + '/tensorflow_logs'
            os.system('tensorboard --logdir=%s' % whole_path)
        elif (path is None) and (not hasattr(self, '_network')):
            raise ValueError('please give path to logs')
        else:
            if path[0] == '/':
                path = path[1:]
            os.system('tensorboard --logdir=%s' % path)

    def _get_feed_dict(self, xs, ys, *args):
        if self._network.reg == 'dropout':
            feed_dict = {self._network.data_placeholder: xs,
                         self._network.target_placeholder: ys,
                         self._network.keep_prob_placeholder: args[0]}
        elif self._network.reg == 'l2':
            feed_dict = {self._network.data_placeholder: xs,
                         self._network.target_placeholder: ys,
                         self._network.l2_placeholder: args[0]}
        else:
            feed_dict = {self._network.data_placeholder: xs,
                         self._network.target_placeholder: ys}
        return feed_dict
