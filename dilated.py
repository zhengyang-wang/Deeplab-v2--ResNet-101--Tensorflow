import tensorflow as tf
import numpy as np
import six
import sys



"""
This script provides different 2d dilated convolutions.

I appreciate ideas for a more efficient implementation of the proposed two smoothed dilated convolutions.
"""



def _dilated_conv2d(dilated_type, x, kernel_size, num_o, dilation_factor, name, biased=False):
	if dilated_type == 'regular':
		return _regular_dilated_conv2d(x, kernel_size, num_o, dilation_factor, name, biased)
	elif dilated_type == 'decompose':
		return _decomposed_dilated_conv2d(x, kernel_size, num_o, dilation_factor, name, biased)
	elif dilated_type == 'smooth_GI':
		return _smoothed_dilated_conv2d_GI(x, kernel_size, num_o, dilation_factor, name, biased)
	elif dilated_type == 'smooth_SSC':
		return _smoothed_dilated_conv2d_SSC(x, kernel_size, num_o, dilation_factor, name, biased)
	else:
		print('dilated_type ERROR!')
		print("Please input: regular, decompose, smooth_GI or smooth_SSC")
		sys.exit(-1)

def _regular_dilated_conv2d(x, kernel_size, num_o, dilation_factor, name, biased=False):
	"""
	Dilated conv2d without BN or relu.
	"""
	num_x = x.shape[3].value
	with tf.variable_scope(name) as scope:
		w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
		o = tf.nn.atrous_conv2d(x, w, dilation_factor, padding='SAME')
		if biased:
			b = tf.get_variable('biases', shape=[num_o])
			o = tf.nn.bias_add(o, b)
		return o

def _decomposed_dilated_conv2d(x, kernel_size, num_o, dilation_factor, name, biased=False):
	"""
	Decomposed dilated conv2d without BN or relu.
	"""
	# padding so that the input dims are multiples of dilation_factor
	H = tf.shape(x)[1]
	W = tf.shape(x)[2]
	pad_bottom = (dilation_factor - H % dilation_factor) if H % dilation_factor != 0 else 0
	pad_right = (dilation_factor - W % dilation_factor) if W % dilation_factor != 0 else 0
	pad = [[0, pad_bottom], [0, pad_right]]
	# decomposition to smaller-sized feature maps
	# [N,H,W,C] -> [N*d*d, H/d, W/d, C]
	o = tf.space_to_batch(x, paddings=pad, block_size=dilation_factor)
	# perform regular conv2d
	num_x = x.shape[3].value
	with tf.variable_scope(name) as scope:
		w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
		s = [1, 1, 1, 1]
		o = tf.nn.conv2d(o, w, s, padding='SAME')
		if biased:
			b = tf.get_variable('biases', shape=[num_o])
			o = tf.nn.bias_add(o, b)
	o = tf.batch_to_space(o, crops=pad, block_size=dilation_factor)
	return o

def _smoothed_dilated_conv2d_GI(x, kernel_size, num_o, dilation_factor, name, biased=False):
	"""
	Smoothed dilated conv2d via the Group Interaction (GI) layer without BN or relu.
	"""
	# padding so that the input dims are multiples of dilation_factor
	H = tf.shape(x)[1]
	W = tf.shape(x)[2]
	pad_bottom = (dilation_factor - H % dilation_factor) if H % dilation_factor != 0 else 0
	pad_right = (dilation_factor - W % dilation_factor) if W % dilation_factor != 0 else 0
	pad = [[0, pad_bottom], [0, pad_right]]
	# decomposition to smaller-sized feature maps
	# [N,H,W,C] -> [N*d*d, H/d, W/d, C]
	o = tf.space_to_batch(x, paddings=pad, block_size=dilation_factor)
	# perform regular conv2d
	num_x = x.shape[3].value
	with tf.variable_scope(name) as scope:
		w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
		s = [1, 1, 1, 1]
		o = tf.nn.conv2d(o, w, s, padding='SAME')
		fix_w = tf.Variable(tf.eye(dilation_factor*dilation_factor), name='fix_w')
		l = tf.split(o, dilation_factor*dilation_factor, axis=0)
		os = []
		for i in six.moves.range(0, dilation_factor*dilation_factor):
			os.append(fix_w[0, i] * l[i])
			for j in six.moves.range(1, dilation_factor*dilation_factor):
				os[i] += fix_w[j, i] * l[j]
		o = tf.concat(os, axis=0)
		if biased:
			b = tf.get_variable('biases', shape=[num_o])
			o = tf.nn.bias_add(o, b)
	o = tf.batch_to_space(o, crops=pad, block_size=dilation_factor)
	return o

def _smoothed_dilated_conv2d_SSC(x, kernel_size, num_o, dilation_factor, name, biased=False):
	"""
	Smoothed dilated conv2d via the Separable and Shared Convolution (SSC) without BN or relu.
	"""
	num_x = x.shape[3].value
	fix_w_size = dilation_factor * 2 - 1
	with tf.variable_scope(name) as scope:
		fix_w = tf.get_variable('fix_w', shape=[fix_w_size, fix_w_size, 1, 1, 1], initializer=tf.zeros_initializer)
		mask = np.zeros([fix_w_size, fix_w_size, 1, 1, 1], dtype=np.float32)
		mask[dilation_factor - 1, dilation_factor - 1, 0, 0, 0] = 1
		fix_w = tf.add(fix_w, tf.constant(mask, dtype=tf.float32))
		o = tf.expand_dims(x, -1)
		o = tf.nn.conv3d(o, fix_w, strides=[1,1,1,1,1], padding='SAME')
		o = tf.squeeze(o, -1)
		w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
		o = tf.nn.atrous_conv2d(o, w, dilation_factor, padding='SAME')
		if biased:
			b = tf.get_variable('biases', shape=[num_o])
			o = tf.nn.bias_add(o, b)
		return o