import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import PIL.Image
import IPython.display
import io
import random
import time
import statistics as stats
from matplotlib import pyplot as plt

class CellularAutomata(tf.keras.Model):
	def __init__(self, img_size: int, 
	channel_count: int, layer_counts: [int], perception_kernel):
		super().__init__()

		self.img_size = img_size
		self.channel_count = channel_count
		self.conserve_mass = False
		self.noise = 0.0
		self.clamp_values = True

		perception_input = tf.keras.layers.Input(shape=(img_size, img_size, channel_count * 3))
		curr_layer = perception_input
		for layer_count in layer_counts:
			curr_layer = tf.keras.layers.Conv2D(filters=layer_count, kernel_size=1,
				activation=tf.nn.relu)(curr_layer)
		output_layer = tf.keras.layers.Conv2D(filters=channel_count, kernel_size=1,
			activation=None, kernel_initializer=tf.zeros_initializer)(curr_layer)

		self.model = tf.keras.Model(inputs=[perception_input], outputs=output_layer)

		# Project the perception tensor so that it is 4D. This is used by the depthwise convolution
		# to create a dot product along the 3rd axis, but we don't need that so we index
		# it with None:
		perception_kernel = perception_kernel[:, :, None, :]
		perception_kernel = tf.repeat(perception_kernel, 
			repeats=self.channel_count, axis=2)
		self.perception_kernel = perception_kernel

	def pad_repeat(self, tensor):
		multiples = [3, 3]
		t1 = tf.tile(tensor, multiples)
		w = len(tensor)
		return t1[w-1 : w-1+w+2, w-1 : w-1+w+2]

	@tf.function
	def perceive(self, x):
		# Pad the input state around the boundaries using the topology of a torus 
		# to make sure that the world's behavior is isotropic.
		multiples = [1, 3, 3, 1]
		t1 = tf.tile(x, multiples)
		w = self.img_size
		x = t1[:, w-1 : w-1+w+2, w-1 : w-1+w+2, :]

		conv = tf.nn.depthwise_conv2d(x, 
			filter=self.perception_kernel, 
			strides=[1, 1, 1, 1],
			padding="VALID")
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
		x += (tf.cast(tf.random.uniform(tf.shape(x[:, :, :, :])), tf.float32) - 0.5) * 2.0 * self.noise
		
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

	def constfilled(self, x):
		""" Fills the world with ones. """
		return np.ones((self.img_size, self.img_size, self.channel_count), dtype=np.float32) * x
			
	def pointfilled(self):
		""" Fills the world with zeros except for a single point. """
		x = np.zeros((self.img_size, self.img_size, self.channel_count), dtype=np.float32)
		x[self.img_size // 2, self.img_size // 2] = np.ones((self.channel_count,))
		return x
	
	def randomfilled(self):
		""" Fills the world with random numbers from 0 to the random fill maximum. """
		x = np.random.rand(self.img_size, self.img_size, self.channel_count).astype(np.float32)
		return x
			
	def to_image(self, x, scale=1):
		# Slice off all the non-color (hidden channels):
		arr = x[..., :3]
		rgb_array = np.uint8(arr * 255.0)

		# Scale the first two dimensions of the image by the given scale.
		for dim in range(2):
				rgb_array = np.repeat(rgb_array, scale, dim)

		out = io.BytesIO()
		return PIL.Image.fromarray(rgb_array)
		PIL.Image.fromarray(rgb_array).save(out, 'png')
		return IPython.display.Image(data=out.getvalue())
	
	def display_gif(self, xs, scale=None):
		if scale is None:
			scale = 64 // self.img_size
				
		out = io.BytesIO()
		imgs = [self.to_image(x, scale) for x in xs]
		durs = [100 for img in imgs]
		durs[0] = 1000
		durs[-1] = 1000
		imgs[0].save(out, 'gif', save_all=True, append_images=imgs[1:], loop=0, duration=durs)
		img = IPython.display.Image(data=out.getvalue())
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
	
	def show_sample_run(self, x0, xf, lifetime):
		print("Target:")
		self.ca.display(xf())
		
		# Run the CA for its lifetime with the current weights.
		x = x0()[None, ...]
				
		xs = []
		for i in range(lifetime):
			xs.append(x[0, ..., :3])
			x = self.ca(x)
		xs.append(x[0, ..., :3])

		print("Sample run:")
		self.ca.display_gif(xs)
			
	def show_loss_history(self):
		if self.loss_hist:
			print("\n step: %d, log10(loss): %.3f" % (
				len(self.loss_hist), np.log10(self.loss_hist[-1])), end='')
			plt.plot(self.loss_hist)
			plt.yscale('log')
			plt.grid()
			plt.show()
	
	def run(self, x0, xf, lifetime, seconds=5):
		initial = result = loss = None
		start = time.time()
		elapsed_seconds = 0.0

		while elapsed_seconds < seconds:
			initial = np.repeat(x0()[None, ...], 1, 0)
			target = np.repeat(xf()[None, ...], 1, 0)
			x, loss = self.train_step(initial, target, lifetime)
			self.loss_hist.append(loss.numpy())
			
			# Feed the final state back in and train again on that.
			#x, loss = self.train_step(x, target, lifetime)
			
			elapsed_seconds = time.time() - start
					
	def save(self):
		self.ca.model.save_weights("./checkpoints/data")
			
	def save_exists(self):
		return os.path.isdir("./checkpoints")
			
	def load(self):
		self.ca.model.load_weights("./checkpoints/data")

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

def init_training(ca, do_load=False, learning_rate=1.0e-3):
	ca.model.summary()
	training = Training(ca=ca, learning_rate=learning_rate)
	if training.save_exists() and do_load:
		training.load()
		print("Loaded trained model")
	return training