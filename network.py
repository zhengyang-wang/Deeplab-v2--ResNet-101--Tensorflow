import tensorflow as tf
import numpy as np
import six



"""
This script defines Deeplab v2 network.
To use the pre-trained ResNet-101, the name of each layer is the same as 
the original version in Caffe.
"""



class DeepLab_v2_Network(object):

	def __init__(self, inputs, num_classes, is_training):
		self.inputs = inputs
		self.num_classes = num_classes
		self.channel_axis = 3

		# For a small batch size, it is better to keep 
		# the statistics of the BN layers (running means and variances)
		# frozen, and to not update the values provided by the pre-trained model. 
		# If is_training=True, the statistics will be updated during the training.
		# Note that is_training=False still updates BN parameters gamma (scale) and beta (offset)
		# if they are presented in var_list of the optimiser definition.
		self.is_training = is_training
		
		self.build_network()

	def build_network(self):
		outputs = self._start_block() # 321->81

		outputs = self._bottleneck_resblock(outputs, 256, '2a',
			identity_connection=False) # 81->81
		outputs = self._bottleneck_resblock(outputs, 256, '2b')
		outputs = self._bottleneck_resblock(outputs, 256, '2c')

		outputs = self._bottleneck_resblock(outputs, 512, '3a',
			half_size=True, identity_connection=False) # 81->41
		for i in six.moves.range(1, 4):
			outputs = self._bottleneck_resblock(outputs, 512, '3b%d' % i)

		outputs = self._dilated_bottle_resblock(outputs, 1024, 2, '4a',
			identity_connection=False) # 41->41
		for i in six.moves.range(1, 23):
			outputs = self._dilated_bottle_resblock(outputs, 1024, 2, '4b%d' % i)

		outputs = self._dilated_bottle_resblock(outputs, 2048, 4, '5a',
			identity_connection=False) # 41->41
		outputs = self._dilated_bottle_resblock(outputs, 2048, 4, '5b')
		outputs = self._dilated_bottle_resblock(outputs, 2048, 4, '5c')

		self.o = self._ASPP(outputs, self.num_classes, [6, 12, 18, 24])

	# blocks
	def _start_block(self):
		outputs = self._conv2d(self.inputs, 7, 64, 2, name='conv1')
		outputs = self._batch_norm(outputs, name='bn_conv1',
			is_training=self.is_training, activation_fn=tf.nn.relu)
		outputs = self._max_pool2d(outputs, 3, 2, name='pool1')
		return outputs

	def _bottleneck_resblock(self, x, num_o, name, half_size=False, identity_connection=True):
		first_s = 2 if half_size else 1
		assert num_o % 4 == 0, 'Bottleneck number of output ERROR!'
		# branch1
		if not identity_connection:
			o_b1 = self._conv2d(x, 1, num_o, first_s, name='res%s_branch1' % name)
			o_b1 = self._batch_norm(o_b1, name='bn%s_branch1' % name,
				is_training=self.is_training, activation_fn=None)
		else:
			o_b1 = x
		# branch2
		o_b2a = self._conv2d(x, 1, num_o / 4, first_s, name='res%s_branch2a' % name)
		o_b2a = self._batch_norm(o_b2a, name='bn%s_branch2a' % name,
			is_training=self.is_training, activation_fn=tf.nn.relu)

		o_b2b = self._conv2d(o_b2a, 3, num_o / 4, 1, name='res%s_branch2b' % name)
		o_b2b = self._batch_norm(o_b2b, name='bn%s_branch2b' % name,
			is_training=self.is_training, activation_fn=tf.nn.relu)

		o_b2c = self._conv2d(o_b2b, 1, num_o, 1, name='res%s_branch2c' % name)
		o_b2c = self._batch_norm(o_b2c, name='bn%s_branch2c' % name,
			is_training=self.is_training, activation_fn=None)
		# add
		outputs = self._add([o_b1,o_b2c], name='res%s' % name)
		# relu
		outputs = self._relu(outputs, name='res%s_relu' % name)
		return outputs

	def _dilated_bottle_resblock(self, x, num_o, dilation_factor, name, identity_connection=True):
		assert num_o % 4 == 0, 'Bottleneck number of output ERROR!'
		# branch1
		if not identity_connection:
			o_b1 = self._conv2d(x, 1, num_o, 1, name='res%s_branch1' % name)
			o_b1 = self._batch_norm(o_b1, name='bn%s_branch1' % name,
				is_training=self.is_training, activation_fn=None)
		else:
			o_b1 = x
		# branch2
		o_b2a = self._conv2d(x, 1, num_o / 4, 1, name='res%s_branch2a' % name)
		o_b2a = self._batch_norm(o_b2a, name='bn%s_branch2a' % name,
			is_training=self.is_training, activation_fn=tf.nn.relu)

		o_b2b = self._dilated_conv2d(o_b2a, 3, num_o / 4, dilation_factor, name='res%s_branch2b' % name)
		o_b2b = self._batch_norm(o_b2b, name='bn%s_branch2b' % name,
			is_training=self.is_training, activation_fn=tf.nn.relu)

		o_b2c = self._conv2d(o_b2b, 1, num_o, 1, name='res%s_branch2c' % name)
		o_b2c = self._batch_norm(o_b2c, name='bn%s_branch2c' % name,
			is_training=self.is_training, activation_fn=None)
		# add
		outputs = self._add([o_b1,o_b2c], name='res%s' % name)
		# relu
		outputs = self._relu(outputs, name='res%s_relu' % name)
		return outputs

	def _ASPP(self, x, num_o, dilations):
		o = []
		for i, d in enumerate(dilations):
			o.append(self._dilated_conv2d(x, 3, num_o, d, name='fc1_voc12_c%d' % i, biased=True))
		return self._add(o, name='fc1_voc12')

	# layers
	def _conv2d(self, x, kernel_size, num_o, stride, name, biased=False):
		"""
		Conv2d without BN or relu.
		"""
		num_x = x.shape[self.channel_axis].value
		with tf.variable_scope(name) as scope:
			w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
			s = [1, stride, stride, 1]
			o = tf.nn.conv2d(x, w, s, padding='SAME')
			if biased:
				b = tf.get_variable('biases', shape=[num_o])
				o = tf.nn.bias_add(o, b)
			return o

	def _dilated_conv2d(self, x, kernel_size, num_o, dilation_factor, name, biased=False):
		"""
		Dilated conv2d without BN or relu.
		"""
		num_x = x.shape[self.channel_axis].value
		with tf.variable_scope(name) as scope:
			w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
			o = tf.nn.atrous_conv2d(x, w, dilation_factor, padding='SAME')
			if biased:
				b = tf.get_variable('biases', shape=[num_o])
				o = tf.nn.bias_add(o, b)
			return o

	def _relu(self, x, name):
		return tf.nn.relu(x, name=name)

	def _add(self, x_l, name):
		return tf.add_n(x_l, name=name)

	def _max_pool2d(self, x, kernel_size, stride, name):
		k = [1, kernel_size, kernel_size, 1]
		s = [1, stride, stride, 1]
		return tf.nn.max_pool(x, k, s, padding='SAME', name=name)

	def _batch_norm(self, x, name, is_training, activation_fn):
		with tf.variable_scope(name) as scope:
			o = tf.contrib.layers.batch_norm(
				x,
				scale=True,
				activation_fn=activation_fn,
				updates_collections=None,
				is_training=is_training,
				scope=scope)
			return o