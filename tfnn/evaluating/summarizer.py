import os
import shutil
import tfnn


class Summarizer(object):
    def __init__(self, network=None, save_path='/tmp', include_test=False):
        self._include_test = include_test
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
            self.train_writer = tfnn.train.SummaryWriter(self.save_path + '/' + self._folder + '/train',
                                                         network.sess.graph)
            if include_test:
                self.test_writer = tfnn.train.SummaryWriter(self.save_path + '/' + self._folder + '/test', )

    def record_train(self, t_xs, t_ys, global_step, *args):
        if self._network.reg == 'dropout':
            if len(args) != 1:
                raise ValueError('Do not find keep_prob value.')
            keep_prob = args[0]
            l2_lambda = 0.
        elif self._network.reg == 'l2':
            if len(args) != 1:
                raise ValueError('Do not find l2_lambda value.')
            keep_prob = 1.
            l2_lambda = args[0]
        else:
            keep_prob, l2_lambda = 1., 0.
        feed_dict = self._get_feed_dict(t_xs, t_ys, keep=keep_prob, l2=l2_lambda)
        train_result = self._network.sess.run(self.merged, feed_dict)
        self.train_writer.add_summary(train_result, global_step)

    def record_test(self, v_xs, v_ys, global_step):
        if not self._include_test:
            raise ReferenceError('Set tfnn.Summarizer(include_test=True) to record test')
        feed_dict = self._get_feed_dict(v_xs, v_ys, keep=1., l2=0.)
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

    def _get_feed_dict(self, xs, ys, keep, l2):
        if self._network.reg == 'dropout':
            feed_dict = {self._network.data_placeholder: xs,
                         self._network.target_placeholder: ys,
                         self._network.keep_prob_placeholder: keep}
        elif self._network.reg == 'l2':
            feed_dict = {self._network.data_placeholder: xs,
                         self._network.target_placeholder: ys,
                         self._network.l2_placeholder: l2}
        else:
            feed_dict = {self._network.data_placeholder: xs,
                         self._network.target_placeholder: ys}
        return feed_dict
