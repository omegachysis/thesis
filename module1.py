import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import PIL.Image
import IPython.display
import math
import io
import random
import time
import statistics as stats
from matplotlib import pyplot as plt

class EdgeStrategy:
	TF_SAME = 0
	TORUS = 1
	ZEROS = 'pad_const zeros'
	ONES = 'pad_const ones'
	MIRROR = 2
	RANDOM = 3

class CellularAutomata(tf.keras.Model):
	def __init__(self, img_size: int, 
	channel_count: int, layer_counts: [int], perception_kernel):
		super().__init__()

		self.img_size = img_size
		self.channel_count = channel_count
		self.conserve_mass = False
		self.noise_range = (0.0, 0.0)
		self.clamp_values = True
		self.edge_strategy = EdgeStrategy.TF_SAME

		# Project the perception tensor so that it is 4D. This is used by the depthwise convolution
		# to create a dot product along the 3rd axis, but we don't need that so we index
		# it with None:
		perception_kernel = perception_kernel[:, :, None, :]
		perception_kernel = tf.repeat(perception_kernel, 
			repeats=self.channel_count, axis=2)
		self.perception_kernel = perception_kernel

		perception_input = tf.keras.layers.Input(
			shape=(img_size, img_size, self.channel_count * perception_kernel.shape[-1]))
		curr_layer = perception_input
		for layer_count in layer_counts:
			curr_layer = tf.keras.layers.Conv2D(filters=layer_count, kernel_size=1,
				activation=tf.nn.relu)(curr_layer)
		output_layer = tf.keras.layers.Conv2D(filters=channel_count, kernel_size=1,
			activation=None, kernel_initializer=tf.zeros_initializer)(curr_layer)

		self.model = tf.keras.Model(inputs=[perception_input], outputs=output_layer)

	def pad_repeat(self, tensor):
		multiples = [3, 3]
		t1 = tf.tile(tensor, multiples)
		w = len(tensor)
		return t1[w-1 : w-1+w+2, w-1 : w-1+w+2]

	@tf.function
	def perceive(self, x):
		pad_mode = None

		if self.edge_strategy == EdgeStrategy.TF_SAME:
			pad_mode = "SAME"

		elif self.edge_strategy == EdgeStrategy.TORUS:
			pad_mode = "VALID"
			# Pad the input state around the boundaries using the topology of a torus 
			# to make sure that the world's behavior is isotropic.
			multiples = [1, 3, 3, 1]
			t1 = tf.tile(x, multiples)
			w = self.img_size
			x = t1[:, w-1 : w-1+w+2, w-1 : w-1+w+2, :]

		elif str(self.edge_strategy).startswith('pad_const '):
			const_val = 0.0
			if self.edge_strategy == EdgeStrategy.ZEROS:
				const_val = 0.0
			elif self.edge_strategy == EdgeStrategy.ONES:
				const_val = 1.0

			pad_mode = "VALID"
			paddings = tf.constant([[0,0], [1,1], [1,1], [0,0]])
			x = tf.pad(x, paddings, "CONSTANT", constant_values=const_val)

		elif self.edge_strategy == EdgeStrategy.MIRROR:
			pad_mode = "VALID"
			paddings = tf.constant([[0,0], [1,1], [1,1], [0,0]])
			x = tf.pad(x, paddings, "SYMMETRIC")

		elif self.edge_strategy == EdgeStrategy.RANDOM:
			pad_mode = "VALID"
			paddings = tf.constant([[0,0], [1,1], [1,1], [0,0]])
			x = tf.pad(x, paddings, "CONSTANT")
			mask = tf.constant(0.0, shape=(1, self.img_size, self.img_size, self.channel_count))
			mask = tf.pad(mask, paddings, "CONSTANT", constant_values=1.0)
			noise = tf.cast(tf.random.uniform(tf.shape(mask[...])), tf.float32)
			x += mask * noise

		conv = tf.nn.depthwise_conv2d(x, 
			filter=self.perception_kernel, 
			strides=[1, 1, 1, 1],
			padding=pad_mode)

		return conv

	@tf.function
	def call(self, x):
		s = self.perceive(x)
		dx = self.model(s)
		
		# Add mass conservation to the model by subtracting the average of the dx values.
		if self.conserve_mass:
			dx -= tf.math.reduce_mean(dx)
		x += dx
		
		# Add random noise.
		noise_len = self.noise_range[1] - self.noise_range[0]
		noise_val = tf.cast(tf.random.uniform(tf.shape(x[:, :, :, :])), tf.float32)
		x += noise_val * noise_len + self.noise_range[0]
		
		# Keep random noise or changes in dx from causing out-of-range values.
		if self.clamp_values:
			x = tf.clip_by_value(x, 0.0, 1.0)
				
		return x
	
	def imagefilled(self, image_path):
		""" Fills the world with image data from the disk. """
		x = self.constfilled(0.0)
		img = PIL.Image.open(image_path).convert("RGB")
		img = img.resize(size=(self.img_size, self.img_size))
		color_arr = np.float32(img) / 255.0
		x[:, :, :3] = color_arr
		return x

	def constfilled(self, u):
		""" Fills the world with u. """
		return np.ones((self.img_size, self.img_size, self.channel_count), dtype=np.float32) * u
			
	def pointfilled(self, x, point_value):
		""" Add a single point of value u. """
		x[self.img_size // 2, self.img_size // 2] = np.ones((self.channel_count,)) * point_value
		return x
	
	def randomfilled(self):
		""" Fills the world with random numbers from 0 to the random fill maximum. """
		x = np.random.rand(self.img_size, self.img_size, self.channel_count).astype(np.float32)
		return x

	def bordered(self, x, border_value):
		""" Fills the input state with a border of the given value. """
		shrunk = x[1:-1, 1:-1, :]
		padded = np.pad(shrunk, pad_width=1, mode='constant', constant_values=border_value)
		return padded[:, :, 1:-1]
			
	def to_image(self, x, scale=1):
		hsize = math.ceil(self.channel_count / 3)
		arr = np.zeros((self.img_size, self.img_size * hsize, 3))
		# Fill the image array with the RGB images generated from the 
		# state in RGB and the hidden channels in groups of 3.
		for i in range(hsize):
			arr[:, self.img_size*i : self.img_size*(i+1), :] = x[..., i*3 : (i+1)*3]

		rgb_array = np.uint8(arr * 255.0)

		# Scale the first two dimensions of the image by the given scale.
		for dim in range(2):
				rgb_array = np.repeat(rgb_array, scale, dim)

		out = io.BytesIO()
		return PIL.Image.fromarray(rgb_array)
		PIL.Image.fromarray(rgb_array).save(out, 'png')
		return IPython.display.Image(data=out.getvalue())
	
	def create_gif(self, xs, scale=None):
		if scale is None:
			scale = 128 // self.img_size
				
		out = io.BytesIO()
		imgs = [self.to_image(x, scale) for x in xs]
		durs = [50 for img in imgs]
		durs[0] = 500
		durs[-1] = 500
		imgs[0].save(out, 'gif', save_all=True, append_images=imgs[1:], loop=0, duration=durs)
		return out.getvalue()
		
	def display_gif(self, xs, scale=None):
		img = IPython.display.Image(data=self.create_gif(xs, scale))
		IPython.display.display(img)

	def display(self, x, scale=None):
		if scale is None:
			scale = 64 // self.img_size
				
		out = io.BytesIO()
		self.to_image(x, scale).save(out, 'png')
		img = IPython.display.Image(data=out.getvalue())
		IPython.display.display(img)

class Training(object):
	def __init__(self, ca, learning_rate):
		self.ca = ca
		self.loss_hist = []
		self.learning_rate = learning_rate
		self.lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
			boundaries = [2000], 
			values = [self.learning_rate, self.learning_rate * 0.1])
		self.trainer = tf.keras.optimizers.Adam(self.lr_sched)

	def get_loss(self, x, target):
		return tf.reduce_mean(tf.square(x[..., :3] - target[..., :3]))

	def get_sum(self, x):
		return tf.reduce_sum(x)

	@tf.function
	def train_step(self, x0, xf, lifetime):
		x = x0
		with tf.GradientTape() as g:
			for i in tf.range(lifetime):
				x = self.ca(x)
			loss = tf.reduce_mean(self.get_loss(x, xf))
				
		grads = g.gradient(loss, self.ca.weights)
		grads = [g / (tf.norm(g) + 1.0e-8) for g in grads]
		self.trainer.apply_gradients(zip(grads, self.ca.weights))
		return x, loss

	def do_sample_run(self, x0, xf, lifetime):
		# Run the CA for its lifetime with the current weights.
		x = x0()[None, ...]
				
		xs = []
		xs.append(x[0, ...])
		for i in range(lifetime):
			x = self.ca(x)
			xs.append(x[0, ...])

		return xs
	
	def show_sample_run(self, x0, xf, lifetime):
		print("Target:")
		self.ca.display(xf())

		xs = self.do_sample_run(x0, xf, lifetime)
	
		print("Sample run:")
		self.ca.display_gif(xs)

	def _graph_loss_hist(self):
		plt.clf()
		if self.loss_hist:
			print("\n step: %d, loss: %.3f, log10(loss): %.3f" % (
				len(self.loss_hist), self.loss_hist[-1], np.log10(self.loss_hist[-1])), end='')
			plt.plot(self.loss_hist)
			plt.yscale('log')
			plt.grid()
			
	def show_loss_history(self):
		if self.loss_hist:
			self._graph_loss_hist()
			plt.show()

	def is_done(self):
		return self.loss_hist and \
			self.loss_hist[-1] * self.ca.img_size * self.ca.img_size * 3 <= 0.001
	
	def run(self, x0, xf, lifetime, max_seconds=None, max_steps=None, target_loss=None,
		max_plateau_len=None):
		if self.is_done(): return

		initial = result = loss = None
		start = time.time()
		elapsed_seconds = 0.0

		best_loss = None
		plateau = 0

		num_steps = 0
		while True:
			if max_steps is not None:
				if num_steps >= max_steps: 
					print("Stopping due to max steps reached")
					return
			if max_seconds is not None:
				if elapsed_seconds >= max_seconds: 
					print("Stopping due to time-out")
					return
			if target_loss is not None:
				if self.loss_hist and self.loss_hist[-1] <= target_loss: 
					print("Stopping due to target loss reached")
					return
			if max_plateau_len is not None:
				if plateau >= max_plateau_len:
					print("Stopping due to plateau")
					return
					
			initial = np.repeat(x0()[None, ...], 1, 0)
			target = np.repeat(xf()[None, ...], 1, 0)
			x, loss = self.train_step(initial, target, lifetime)
			if best_loss is None or loss.numpy() < best_loss:
				best_loss = loss.numpy()
				plateau = 0
			else:
				plateau += 1

			self.loss_hist.append(loss.numpy())
			# Feed the final state back in and train again on that.
			x, loss = self.train_step(x, target, lifetime)
			elapsed_seconds = time.time() - start
			
			num_steps += 1
			if self.is_done(): 
				print("Stopping due to zero loss")
				return

		else:
			raise ValueError()
					
	def save(self, name, sample_run_xs):
		self.ca.model.save_weights(f"./results/{name}_weights")
		with open(f"./results/{name}_loss_hist.txt", 'w') as f:
			f.writelines([str(loss)+'\n' for loss in self.loss_hist])
		self._graph_loss_hist()
		plt.savefig(f"./results/{name}_loss_hist.png")
		with open(f"./results/{name}_sample_run.gif", 'wb') as f:
			f.write(self.ca.create_gif(sample_run_xs))

def sobel_state_kernel():
	# Create a Sobel filter:
	identity = np.float32([0, 1, 0])
	identity = np.outer(identity, identity)
	dx = np.outer(np.float32([1, 2, 1]), np.float32([-1, 0, 1])) / 8.0
	dy = dx.T
	return tf.stack([identity, dx - dy, dx + dy], axis=-1)

def tensor_basis_kernel():
	basis = []
	for x in range(3):
		for y in range(3):
			tensor = np.zeros((3, 3), dtype=np.float32)
			tensor[x, y] = 1
			basis.append(tensor)
	return tf.stack(basis, axis=-1)

def init_training(ca, learning_rate=1.0e-3):
	ca.model.summary()
	training = Training(ca=ca, learning_rate=learning_rate)
	return training