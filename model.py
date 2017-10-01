from datetime import datetime
import os
import sys
import time
import numpy as np
import tensorflow as tf

from network import DeepLab_v2_Network
from utils import ImageReader, decode_labels, inv_preprocess, prepare_label, write_log



"""
This script trains or evaluates the model on augmented PASCAL VOC 2012 dataset.
The training set contains 10581 training images.
The validation set contains 1449 validation images.

Training:
'poly' learning rate
different learning rates for different layers
"""



IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

class Model(object):

	def __init__(self, sess, conf):
		self.sess = sess
		self.conf = conf

	# train
	def train(self):
		self.train_setup()
		self.train_summary()

		self.sess.run(tf.global_variables_initializer())

		# Load the pre-trained model if provided
		if self.conf.pretrain_file is not None:
			self.load(self.loader, self.conf.pretrain_file)

		# Start queue threads.
		threads = tf.train.start_queue_runners(coord=self.coord, sess=self.sess)

		# Train!
		for step in range(self.conf.num_steps+1):
			start_time = time.time()
			feed_dict = { self.curr_step : step }

			if step % self.conf.save_interval == 0:
				loss_value, images, labels, preds, summary, _ = self.sess.run(
					[self.reduced_loss,
					self.image_batch,
					self.label_batch,
					self.pred,
					self.total_summary,
					self.train_op],
					feed_dict=feed_dict)
				self.summary_writer.add_summary(summary, step)
				self.save(self.saver, step)
			else:
				loss_value, _ = self.sess.run([self.reduced_loss, self.train_op],
					feed_dict=feed_dict)

			duration = time.time() - start_time
			print('step {:d} \t loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, duration))
			write_log('{:d}, {:.3f}'.format(step, loss_value), self.conf.logfile)

		# finish
		self.coord.request_stop()
		self.coord.join(threads)

	# evaluate
	def test(self):
		self.test_setup()

		self.sess.run(tf.global_variables_initializer())
		self.sess.run(tf.local_variables_initializer())

		# load checkpoint
		checkpointfile = self.conf.modeldir+ '/model.ckpt-' + str(self.conf.valid_step)
		self.load(self.loader, checkpointfile)

		# Start queue threads.
		threads = tf.train.start_queue_runners(coord=self.coord, sess=self.sess)

		# Test!
		for step in range(self.conf.valid_num_steps):
			preds, _, _ = self.sess.run([self.pred, self.accu_update_op, self.mIou_update_op])
			if step % 100 == 0:
				print('step {:d}'.format(step))
		print('Pixel Accuracy: {:.3f}'.format(self.accu.eval(session=self.sess)))
		print('Mean IoU: {:.3f}'.format(self.mIoU.eval(session=self.sess)))

		# finish
		self.coord.request_stop()
		self.coord.join(threads)

	def train_setup(self):
		tf.set_random_seed(self.conf.random_seed)
		
		# Create queue coordinator.
		self.coord = tf.train.Coordinator()

		# Input size
		self.input_size = (self.conf.input_height, self.conf.input_width)
		
		# Load reader
		with tf.name_scope("create_inputs"):
			reader = ImageReader(
				self.conf.data_dir,
				self.conf.data_list,
				self.input_size,
				self.conf.random_scale,
				self.conf.random_mirror,
				self.conf.ignore_label,
				IMG_MEAN,
				self.coord)
			self.image_batch, self.label_batch = reader.dequeue(self.conf.batch_size)
		
		# Create network
		net = DeepLab_v2_Network(self.image_batch, num_classes=self.conf.num_classes,
			is_training=self.conf.is_training)
		
		# Network raw output
		self.raw_output = net.o # [batch_size, 41, 41, 21]

		# Output size
		output_size = (self.raw_output.shape[1].value, self.raw_output.shape[2].value)
		
		# Variables that load from pre-trained model.
		# For training, last few layers should not be loaded.
		restore_var = [v for v in tf.global_variables() if 'fc' not in v.name]
		
		# Trainable Variables
		# Note that is_training=False still updates BN parameters gamma (scale) and beta (offset)
		# if they are presented in var_list of the optimiser definition.
		# So we remove them from the list.
		all_trainable = [v for v in tf.trainable_variables() if 'beta' not in v.name and 'gamma' not in v.name]
		# Fine-tune part
		conv_trainable = [v for v in all_trainable if 'fc' not in v.name] # lr * 1.0
		# ASPP part
		fc_trainable = [v for v in all_trainable if 'fc' in v.name]
		fc_w_trainable = [v for v in fc_trainable if 'weights' in v.name] # lr * 10.0
		fc_b_trainable = [v for v in fc_trainable if 'biases' in v.name] # lr * 20.0
		# check
		assert(len(all_trainable) == len(fc_trainable) + len(conv_trainable))
		assert(len(fc_trainable) == len(fc_w_trainable) + len(fc_b_trainable))

		# Groud Truth: ignoring all labels greater or equal than n_classes
		label_proc = prepare_label(self.label_batch, output_size, num_classes=self.conf.num_classes, one_hot=False) # [batch_size, 41, 41]
		raw_gt = tf.reshape(label_proc, [-1,])
		indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, self.conf.num_classes - 1)), 1)
		gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
		raw_prediction = tf.reshape(self.raw_output, [-1, self.conf.num_classes])
		prediction = tf.gather(raw_prediction, indices)

		# Pixel-wise softmax_cross_entropy loss
		loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
		# L2 regularization
		l2_losses = [self.conf.weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]
		# Loss function
		self.reduced_loss = tf.reduce_mean(loss) + tf.add_n(l2_losses)

		# Define optimizers
		# 'poly' learning rate
		base_lr = tf.constant(self.conf.learning_rate)
		self.curr_step = tf.placeholder(dtype=tf.float32, shape=())
		learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - self.curr_step / self.conf.num_steps), self.conf.power))
		# We have several optimizers here in order to handle the different lr_mult
		# which is a kind of parameters in Caffe. This controls the actual lr for each
		# layer.
		opt_conv = tf.train.MomentumOptimizer(learning_rate, self.conf.momentum)
		opt_fc_w = tf.train.MomentumOptimizer(learning_rate * 10.0, self.conf.momentum)
		opt_fc_b = tf.train.MomentumOptimizer(learning_rate * 20.0, self.conf.momentum)
		# To make sure each layer gets updated by different lr's, we do not use 'minimize' here.
		# Instead, we separate the steps compute_grads+update_params.
		# Compute grads
		grads = tf.gradients(self.reduced_loss, conv_trainable + fc_w_trainable + fc_b_trainable)
		grads_conv = grads[:len(conv_trainable)]
		grads_fc_w = grads[len(conv_trainable) : (len(conv_trainable) + len(fc_w_trainable))]
		grads_fc_b = grads[(len(conv_trainable) + len(fc_w_trainable)):]
		# Update params
		train_op_conv = opt_conv.apply_gradients(zip(grads_conv, conv_trainable))
		train_op_fc_w = opt_fc_w.apply_gradients(zip(grads_fc_w, fc_w_trainable))
		train_op_fc_b = opt_fc_b.apply_gradients(zip(grads_fc_b, fc_b_trainable))
		# Finally, get the train_op!
		self.train_op = tf.group(train_op_conv, train_op_fc_w, train_op_fc_b)

		# Saver for storing checkpoints of the model
		self.saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=0)

		# Loader for loading the pre-trained model
		self.loader = tf.train.Saver(var_list=restore_var)

	def train_summary(self):
		# Processed predictions: for visualisation.
		raw_output_up = tf.image.resize_bilinear(self.raw_output, self.input_size)
		raw_output_up = tf.argmax(raw_output_up, axis=3)
		self.pred = tf.expand_dims(raw_output_up, dim=3)

		# Image summary.
		images_summary = tf.py_func(inv_preprocess, [self.image_batch, 2, IMG_MEAN], tf.uint8)
		labels_summary = tf.py_func(decode_labels, [self.label_batch, 2, self.conf.num_classes], tf.uint8)
		preds_summary = tf.py_func(decode_labels, [self.pred, 2, self.conf.num_classes], tf.uint8)

		self.total_summary = tf.summary.image('images',
			tf.concat(axis=2, values=[images_summary, labels_summary, preds_summary]),
			max_outputs=2) # Concatenate row-wise.

		if not os.path.exists(self.conf.logdir):
			os.makedirs(self.conf.logdir)
		self.summary_writer = tf.summary.FileWriter(self.conf.logdir,
			graph=tf.get_default_graph())

	def test_setup(self):
		# Create queue coordinator.
		self.coord = tf.train.Coordinator()

		# Load reader
		with tf.name_scope("create_inputs"):
			reader = ImageReader(
				self.conf.data_dir,
				self.conf.valid_data_list,
				None, # the images have different sizes
				False, # no data-aug
				False, # no data-aug
				self.conf.ignore_label,
				IMG_MEAN,
				self.coord)
			image, label = reader.image, reader.label # [h, w, 3 or 1]
		# Add one batch dimension [1, h, w, 3 or 1]
		self.image_batch, self.label_batch = tf.expand_dims(image, dim=0), tf.expand_dims(label, dim=0)
		
		# Create network
		net = DeepLab_v2_Network(self.image_batch, num_classes=self.conf.num_classes, is_training=False)

		# predictions
		raw_output = net.o # [batch_size, 41, 41, 21]
		raw_output = tf.image.resize_bilinear(raw_output, tf.shape(self.image_batch)[1:3,])
		raw_output = tf.argmax(raw_output, axis=3)
		pred = tf.expand_dims(raw_output, dim=3)
		self.pred = tf.reshape(pred, [-1,])
		# labels
		gt = tf.reshape(self.label_batch, [-1,])
		# Ignoring all labels greater than or equal to n_classes.
		temp = tf.less_equal(gt, self.conf.num_classes - 1)
		weights = tf.cast(temp, tf.int32)

		# fix for tf 1.3.0
		gt = tf.where(temp, gt, tf.cast(temp, tf.uint8))

		# Pixel accuracy
		self.accu, self.accu_update_op = tf.contrib.metrics.streaming_accuracy(
			self.pred, gt, weights=weights)

		# mIoU
		self.mIoU, self.mIou_update_op = tf.contrib.metrics.streaming_mean_iou(
			self.pred, gt, num_classes=self.conf.num_classes, weights=weights)

		# Loader for loading the checkpoint
		self.loader = tf.train.Saver(var_list=tf.global_variables())

	def save(self, saver, step):
		'''
		Save weights.
		'''
		model_name = 'model.ckpt'
		checkpoint_path = os.path.join(self.conf.modeldir, model_name)
		if not os.path.exists(self.conf.modeldir):
			os.makedirs(self.conf.modeldir)
		saver.save(self.sess, checkpoint_path, global_step=step)
		print('The checkpoint has been created.')

	def load(self, saver, filename):
		'''
		Load trained weights.
		''' 
		saver.restore(self.sess, filename)
		print("Restored model parameters from {}".format(filename))
