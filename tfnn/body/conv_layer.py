import tfnn
from tfnn.body.layer import Layer


class PoolingLayer(object):
    def __init__(self, pooling='max', strides=(2, 2), ksize=(2, 2), padding='SAME', name=None):
        self.pooling = pooling
        self.strides = strides
        self.ksize = ksize
        self.padding = padding
        self.name = name

    def pool(self, image, layer_size, n_filters):
        # stride [1, x_movement, y_movement, 1]
        k_x, k_y = self.ksize[0], self.ksize[1]
        stride_x, stride_y = self.strides[0], self.strides[1]
        if self.pooling == 'max':
            self.output = tfnn.nn.max_pool(
                value=image, ksize=[1, k_x, k_y, 1],
                strides=[1, stride_x, stride_y, 1], padding=self.padding)
        elif self.pooling == 'average':
            self.output = tfnn.nn.avg_pool(
                value=image, ksize=[1, k_x, k_y, 1],
                strides=[1, stride_x, stride_y, 1], padding=self.padding)
        else:
            raise ValueError('Not support %s pooling' % self.pooling)

        length = layer_size[0] / stride_x
        width = layer_size[1] / stride_y
        features = n_filters
        if not (type(length) == int) and (type(width) == int):
            raise ValueError('pooling dimension error')
        else:
            self.out_size = [int(length), int(width), features]
        return [self.output, self.out_size]


class ConvLayer(Layer):
    def __init__(self,
                 patch_x, patch_y, n_filters, activator=None,
                 strides=(1, 1), padding='SAME',
                 pooling='max', pool_strides=(2, 2), pool_k=(2, 2),
                 pool_padding='SAME', image_shape=None,
                 dropout_layer=False, w_initial='xavier', name=None,):
        super(ConvLayer, self).__init__(activator, dropout_layer, w_initial,
                                        name, layer_type='conv')
        self._check_activator(activator)
        self.patch_x = patch_x
        self.patch_y = patch_y
        self.n_filters = n_filters
        self.padding = padding
        self.strides = strides
        self.pooling = pooling
        self.pool_strides = pool_strides
        self.pool_k = pool_k
        self.pool_padding = pool_padding
        self.image_shape = image_shape
        self.pooling_layer = PoolingLayer(
                pooling=self.pooling,
                strides=self.pool_strides,
                ksize=self.pool_k,
                padding=self.pool_padding,
                )

        self._params = {
                'patch_x': self.patch_x, 'patch_y': self.patch_y, 'n_filters': self.n_filters,
                'activator': self.activator_name, 'strides': self.strides,
                'padding': self.padding, 'pooling': self.pooling,
                'pool_strides': self.pool_strides, 'pool_k': self.pool_k,
                'pool_padding': self.pool_padding, 'dropout_layer': self.dropout_layer,
                'image_shape': self.image_shape, 'w_initial': self.w_initial, 'name': self.name,
                 }

    def construct(self, layers_configs, layers_results):
        self._check_image_shape(layers_configs, layers_results)
        self.name = self._check_name(layers_configs)
        # in conv, the _in_size should be the [length, width, channels]
        _in_size = layers_configs['neural_structure'][-1]['output_size']
        with tfnn.variable_scope(self.name):
            with tfnn.variable_scope('weights') as weights_scope:
                self.W = self._weight_variable([
                    self.patch_x,  # patch length
                    self.patch_y,  # patch width
                    _in_size[-1],  # filter height / channels
                    self.n_filters
                ],
                    self.w_initial)  # number of filters
                tfnn.histogram_summary(self.name + '/weights', self.W)

                # the image summary for visualizing filters
                weights_scope.reuse_variables()
                weights = tfnn.get_variable('weights', trainable=False)
                # scale weights to [0 255] and convert to uint8 (maybe change scaling?)
                x_min = tfnn.reduce_min(weights)
                x_max = tfnn.reduce_max(weights)
                weights_0_to_1 = (weights - x_min) / (x_max - x_min)
                weights_0_to_255_uint8 = tfnn.image.convert_image_dtype(weights_0_to_1, dtype=tfnn.uint8)
                # to tf.image_summary format [batch_size, height, width, channels]
                W_transposed = tfnn.transpose(weights_0_to_255_uint8, [3, 0, 1, 2])
                # image Tensor must be 4-D with last dim 1, 3, or 4,
                # (n_filter, length, width, channel)
                channels_to_look = 3
                if W_transposed._shape[-1] > channels_to_look:
                    n_chunks = int(W_transposed._shape[-1] // channels_to_look)
                    W_transposed = tfnn.split(3, n_chunks,
                                              W_transposed[:, :, :, :n_chunks * channels_to_look])[0]
                # this will display random 5 filters from the n_filters in conv
                tfnn.image_summary(self.name + '/filters',
                                   W_transposed, max_images=10)

            with tfnn.variable_scope('biases'):
                self.b = self._bias_variable([self.n_filters, ])
                tfnn.histogram_summary(self.name + '/biases', self.b)

            with tfnn.name_scope('Wx_plus_b'):
                product = tfnn.nn.conv2d(
                    input=layers_results['final'][-1],
                    filter=self.W,
                    strides=[1, self.strides[0], self.strides[1], 1],
                    padding=self.padding) \
                    + self.b

            if self.activator is None:
                activated_product = product
            else:
                activated_product = self.activator(product)
            tfnn.histogram_summary(self.name + '/activated_product', activated_product)

        # pooling process
        with tfnn.name_scope('pooling'):
            pooled_product, _out_size = self.pooling_layer.pool(
                image=activated_product, layer_size=_in_size, n_filters=self.n_filters)
            tfnn.histogram_summary(self.name + '/pooled_product', pooled_product)

        _do_dropout = layers_configs['params'][0]['do_dropout']
        if _do_dropout and self.dropout_layer:
            _keep_prob = layers_results['reg_value']
            dropped_product = tfnn.nn.dropout(
                pooled_product,
                _keep_prob,
                name='dropout')
            final_product = dropped_product         # don't have to rescale it back, tf dropout has done this
        else:
            dropped_product = None
            final_product = pooled_product

        self.configs_dict = {
            'type': 'conv',
            'name': self.name,
            'neural_structure': {'input_size': _in_size, 'output_size': _out_size},
            'params': self._params,
        }
        self.results_dict = {
            'Layer': self,
            'Wx_plus_b': product,
            'activated': activated_product,
            'dropped': dropped_product,
            'final': final_product}

    def _check_image_shape(self, layers_configs, layers_results):
        """
        have effect only on the first conv layer
        """
        if self.image_shape is not None:
            if len(layers_configs['type']) == 1:
                if isinstance(self.image_shape, tuple):
                    self.image_shape = list(self.image_shape)
                elif not isinstance(self.image_shape, list):
                    raise ValueError('image_shape can only be a tuple or list')

                # image shape have to be (x, y, channel)
                layers_configs['neural_structure'][-1]['output_size'] = self.image_shape
                _xs_placeholder = layers_results['final'][-1]
                replaced_image_shape = self.image_shape.copy()
                replaced_image_shape.insert(0, -1)
                with tfnn.name_scope('reshape_inputs'):
                    _image_placeholder = tfnn.reshape(_xs_placeholder, replaced_image_shape)
                layers_results['activated'][-1] = _image_placeholder
                layers_results['final'][-1] = _image_placeholder
            else:
                raise IndexError('This is not the first conv layer, leave image_shape as default')
