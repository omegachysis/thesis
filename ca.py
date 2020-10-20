import tensorflow as tf
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

class CellularAutomata(tf.keras.Model):
	def __init__(self, img_size: int, 
	channel_count: int, layer_counts: List[int], perception_kernel, num_subnetworks: int,
	combiner_layer_size: int):
		super().__init__()

		self.img_size = img_size
		self.channel_count = channel_count
		self.conserved_mass = None
		self.noise_range = (0.0, 0.0)
		self.noise_mask = None
		self.noise_replace = False
		self.clamp_values = False
		self.edge_strategy = EdgeStrategy.TF_SAME
		self.lock_map = None

		# Project the perception tensor so that it is 4D. This is used by the depthwise convolution
		# to create a dot product along the 3rd axis, but we don't need that so we index
		# it with None:
		perception_kernel = perception_kernel[:, :, None, :]
		perception_kernel = tf.repeat(perception_kernel, 
			repeats=self.channel_count, axis=2)
		self.perception_kernel = perception_kernel

		# Create the input layer.
		inputs = tf.keras.Input(
			shape=(img_size, img_size, self.channel_count * perception_kernel.shape[-1]), 
			dtype=tf.float32)
		
		# Create sub-networks and sub-models.
		self.submodels: List[tf.keras.Model] = []
		suboutputs = [inputs]

		for _ in range(num_subnetworks):
			# Add a convolutional layer for each of the layer counts specified in the config.
			curr_inputs = inputs
			for layer_count in layer_counts:
				conv_layer = tf.keras.layers.Conv2D(
					filters=layer_count, kernel_size=1, activation=tf.nn.relu)
				curr_inputs = conv_layer(curr_inputs)

			# Create the output layer.
			output_layer = tf.keras.layers.Conv2D(filters=channel_count, kernel_size=1,
				activation=None, kernel_initializer=tf.zeros_initializer())
			outputs = output_layer(curr_inputs)

			suboutputs.append(outputs)
			self.submodels.append(tf.keras.Model(inputs=inputs, outputs=outputs))

		# If there is only one subnetwork, just make the whole network the subnetwork:
		if len(self.submodels) == 1:
			self.model = self.submodels[0]
		else:
			# Else, combine together the subnetworks by adding a convolutional relu before 
			# a final output.
			combined_outputs = tf.keras.layers.concatenate(suboutputs)
			combiner_layer = tf.keras.layers.Conv2D(filters=combiner_layer_size, kernel_size=1,
				activation=tf.nn.relu)
			final_output_layer = tf.keras.layers.Conv2D(filters=channel_count, kernel_size=1,
				activation=None, kernel_initializer=tf.zeros_initializer())
			outputs = final_output_layer(combiner_layer(combined_outputs))
			self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

	def load_into_submodel(self, submodel_idx: int, sub_ca):
		model = self.submodels[submodel_idx]
		model.set_weights(sub_ca.model.get_weights())

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
		x += dx
		
		# if self.lock_map is not None and not lock_release:
		# 	x += dx * self.lock_map
		# else:
		# 	x += dx

		# if self.noise_range is not None:
		# 	# Add random noise.
		# 	noise_len = self.noise_range[1] - self.noise_range[0]
		# 	noise_val = tf.cast(tf.random.uniform(tf.shape(x)), tf.float32)
		# 	if self.noise_mask is not None:
		# 		noise_val *= self.noise_mask

		# 	if self.noise_replace and self.noise_mask is not None:
		# 		x = x * (1.0-self.noise_mask) + \
		# 			noise_val * noise_len + self.noise_range[0]
		# 	else:
		# 		x += noise_val * noise_len + self.noise_range[0]
				
		# if self.clamp_values:
		# 	x = tf.clip_by_value(x, 0., 1.)

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
		return np.ones((self.img_size, self.img_size, self.channel_count), dtype=np.float32) * u
			
	def pointfilled(self, x, point_value, pos=(.5,.5)):
		""" Add a single point of value u. """
		x[int(self.img_size*pos[1]), int(self.img_size*pos[0])] = \
			np.ones((self.channel_count,)) * point_value
		return x

	def pointsfilled(self, x, point_value, positions):
		for pos in positions:
			x = self.pointfilled(x, point_value, pos)
		return x
	
	def randomfilled(self):
		""" Fills the world with random numbers from 0 to the random fill maximum. """
		x = np.random.rand(self.img_size, self.img_size, self.channel_count).astype(np.float32)
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

			for channel in range(self.channel_count):
				x[px, py, channel] = line_value
		return x
			
	def to_image(self, x, scale=1):
		hsize = math.ceil(self.channel_count / 3)
		arr = np.zeros((self.img_size, self.img_size * hsize, 3))
		# Fill the image array with the RGB images generated from the 
		# state in RGB and the hidden channels in groups of 3.
		for i in range(self.channel_count):
			s = x[..., i]
			if i >= 3:
				# Hidden channel, scale to fit color space.
				a = np.min(s)
				b = np.max(s)
				ba = max(1.0, b-a)
				s -= a
				s *= 1/ba
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