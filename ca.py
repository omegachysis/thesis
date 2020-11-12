import tensorflow as tf
from tensorflow import keras
import numpy as np
import PIL.Image
import IPython.display
import math
import io
from typing import List

class EdgeStrategy:
	TF_SAME = 0
	TORUS = 1
	ZEROS = 'pad_const zeros'
	ONES = 'pad_const ones'
	MIRROR = 2
	RANDOM = 3

class CellularAutomata(keras.Model):
	def __init__(self, config, perception_kernel):
		super().__init__()

		self.img_size = config.size
		self.num_channels = config.num_channels
		self.edge_strategy = eval(config.edge_strategy)
		self.hidden_layer_size = config.layer1_size

		if perception_kernel is not None:
			# Project the perception tensor so that it is 4D. This is used by the depthwise convolution
			# to create a dot product along the 3rd axis, but we don't need that so we index
			# it with None:
			perception_kernel = perception_kernel[:, :, None, :]
			perception_kernel = tf.repeat(perception_kernel, 
				repeats=self.num_channels, axis=2)
			self.perception_kernel = perception_kernel
			self.use_perception_model = False
			self.perception_model = None
		else:
			self.perception_model = keras.Sequential([
				keras.layers.Conv2D(config.perceive_layer_size, 
					kernel_size=config.perception_kernel_size, activation=tf.nn.relu, padding="SAME"),
			])
			self.use_perception_model = True

		self.model = keras.Sequential([
			keras.layers.Conv2D(self.hidden_layer_size, kernel_size=1, activation=tf.nn.relu),
			keras.layers.Conv2D(self.num_channels, kernel_size=1, activation=None)
		])

		# Compile the model:
		self(tf.zeros([1, 3, 3, self.num_channels]))
		self.model.summary()

	def copy_weights_from(self, other):
		self.model.set_weights(other.model.get_weights())

	@staticmethod
	def laplacian(x):
		Δ = tf.reshape(tf.constant([
				[1/4, 1/2, 1/4],
				[1/2, -3,  1/2],
				[1/4, 1/2, 1/4]
		]), shape=[3,3,1])
		Δ = Δ[:,:,None,:]
		channel_count = x.shape[3]
		Δ = tf.repeat(Δ, repeats=channel_count, axis=2)
		Δx = tf.nn.depthwise_conv2d(x, Δ, strides=[1,1,1,1], padding="SAME")[0]
		return Δx

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
			mask = tf.constant(0.0, shape=(1, self.img_size, self.img_size, self.num_channels))
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
		if self.use_perception_model:
			s = self.perception_model(x)
		else:
			s = self.perceive(x)

		dx = self.model(s)
		x += dx
		return x
	
	def imagefilled(self, image_path):
		""" Fills the world with image data from the disk. """
		x = self.constfilled(1.0)
		img = PIL.Image.open(image_path).convert("RGB")
		img = img.resize(size=(self.img_size, self.img_size))
		color_arr = np.float32(img) / 255.0
		x[:, :, :3] = color_arr
		return x

	def imagestackfilled(self, images: List[str]):
		""" Fills the world with image data from the disk. """
		x = self.constfilled(1.0)
		for i in range(len(images)):
			image_path = images[i]
			img = PIL.Image.open(image_path).convert("RGB")
			img = img.resize(size=(self.img_size, self.img_size))
			color_arr = np.float32(img) / 255.0
			x[..., i*3:i*3+3] = color_arr
		return x

	def constfilled(self, u):
		""" Fills the world with u. """
		return np.ones((self.img_size, self.img_size, self.num_channels), dtype=np.float32) * u
			
	def pointfilled(self, x, point_value, pos=(.5,.5)):
		""" Add a single point of value u. """
		x[int(self.img_size*pos[1]), int(self.img_size*pos[0])] = \
			np.ones((self.num_channels,)) * point_value
		return x

	def pointsfilled(self, x, point_value, positions):
		for pos in positions:
			x = self.pointfilled(x, point_value, pos)
		return x
	
	def randomfilled(self):
		""" Fills the world with random numbers from 0 to the random fill maximum. """
		x = np.random.rand(self.img_size, self.img_size, self.num_channels).astype(np.float32)
		return x

	def bordered(self, x, border_value, width=1):
		""" Fills the input state with a border of the given value. """
		shrunk = x[width:-width, width:-width, :]
		padded = np.pad(shrunk, pad_width=width, mode='constant', constant_values=border_value)
		return padded[:, :, width:-width]

	def linefilled(self, x, line_value, a, b):
		""" Fills the input state with a line from pt a to pt b. """
		ax, ay = a
		bx, by = b
		for i in range(self.img_size*2 + 1):
			t = i/(self.img_size*2)
			px = int((ax + (bx - ax) * t) * self.img_size)
			py = int((ay + (by - ay) * t) * self.img_size)
			if px < 0 or py < 0: continue
			if px >= self.img_size or py >= self.img_size: continue

			for channel in range(self.num_channels):
				x[px, py, channel] = line_value
		return x
			
	def to_image(self, x, scale=1):
		hsize = math.ceil(self.num_channels / 3)
		arr = np.zeros((self.img_size, self.img_size * hsize, 3))
		# Fill the image array with the RGB images generated from the 
		# state in RGB and the hidden channels in groups of 3.
		for i in range(self.num_channels):
			s = x[..., i]
			s = np.clip(s, 0.0, 1.0)
			arr[:, self.img_size*(i//3) : self.img_size*(i//3+1), i%3] = s

		rgb_array = np.uint8(arr * 255.0)
		for dim in range(2):
				rgb_array = np.repeat(rgb_array, scale, dim)
		return PIL.Image.fromarray(rgb_array)
	
	def create_gif(self, xs):
		out = io.BytesIO()
		imgs = [self.to_image(x, scale=3) for x in xs]
		durs = [50 for _ in imgs]
		durs[0] = 500
		durs[-1] = 500
		imgs[0].save(out, 'gif', save_all=True, append_images=imgs[1:], loop=0, duration=durs)
		return out.getvalue()
		
	def display_gif(self, xs):
		img = IPython.display.Image(data=self.create_gif(xs))
		IPython.display.display(img)

	def display(self, x):
		out = io.BytesIO()
		self.to_image(x).save(out, 'png')
		img = IPython.display.Image(data=out.getvalue())
		IPython.display.display(img)