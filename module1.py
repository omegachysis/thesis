import os
import tensorflow as tf
import numpy as np
import PIL.Image
import IPython.display
import math
import io
import time
import random
import wandb
from matplotlib import pyplot as plt
from typing import List

class EdgeStrategy:
	TF_SAME = 0
	TORUS = 1
	ZEROS = 'pad_const zeros'
	ONES = 'pad_const ones'
	MIRROR = 2
	RANDOM = 3

def loss_mean_square(x, target, mask=None):
	if mask is not None:
		return tf.reduce_mean(mask[...,:3] * tf.square(x[...,:3] - target[...,:3]))
	else:
		return tf.reduce_mean(tf.square(x[...,:3] - target[...,:3]))

def loss_all_channels(x, target):
	return tf.reduce_mean(tf.square(x - target))

def loss_harmonize(x):
	channel_count = x.shape[3]
	Δ = tf.reshape(tf.constant([
			[1/4, 1/2, 1/4],
			[1/2, -3,  1/2],
			[1/4, 1/2, 1/4]
	]), shape=[3,3,1])
	Δ = Δ[:,:,None,:]
	Δ = tf.repeat(Δ, repeats=channel_count, axis=2)
	Δx = tf.nn.depthwise_conv2d(x, Δ, strides=[1,1,1,1], padding="VALID")[0]
	return tf.reduce_mean(tf.square(Δx))

class CellularAutomata(tf.keras.Model):
	def __init__(self, img_size: int, 
	channel_count: int, layer_counts: List[int], perception_kernel):
		super().__init__()

		self.img_size = img_size
		self.channel_count = channel_count
		self.conserved_mass = None
		self.noise_range = (0.0, 0.0)
		self.noise_mask = None
		self.noise_replace = False
		self.clamp_values = True
		self.edge_strategy = EdgeStrategy.TF_SAME
		self.lock_map = None

		# Project the perception tensor so that it is 4D. This is used by the depthwise convolution
		# to create a dot product along the 3rd axis, but we don't need that so we index
		# it with None:
		perception_kernel = perception_kernel[:, :, None, :]
		perception_kernel = tf.repeat(perception_kernel, 
			repeats=self.channel_count, axis=2)
		self.perception_kernel = perception_kernel

		# Build the model:
		self.model = tf.keras.Sequential()
		self.model.add(tf.keras.layers.Input(
			shape=(img_size, img_size, self.channel_count * perception_kernel.shape[-1])))
		for layer_count in layer_counts:
			self.model.add(tf.keras.layers.Conv2D(
				filters=layer_count, kernel_size=1, activation=tf.nn.relu))
		self.model.add(tf.keras.layers.Conv2D(filters=channel_count, kernel_size=1,
			activation=None, kernel_initializer=tf.zeros_initializer()))

	def laplacian(self, x):
		Δ = tf.reshape(tf.constant([
				[1/4, 1/2, 1/4],
				[1/2, -3,  1/2],
				[1/4, 1/2, 1/4]
		]), shape=[3,3,1])
		Δ = Δ[:,:,None,:]
		Δ = tf.repeat(Δ, repeats=self.channel_count, axis=2)
		Δx = tf.nn.depthwise_conv2d(x[None,...], Δ, strides=[1,1,1,1], padding="SAME")[0]
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

	def get_mass(self, x):
		return tf.reduce_sum(x, axis=[0,1])

	@tf.function
	def call(self, x, lock_release):
		s = self.perceive(x)
		dx = self.model(s)

		if self.lock_map is not None and not lock_release:
			x += dx * self.lock_map
		else:
			x += dx

		x = tf.clip_by_value(x, 0.0, 1.0)

		if self.conserved_mass is not None:
			new_mass = tf.reduce_sum(x, [1,2])
			x *= (self.conserved_mass + 1e-10) / (new_mass + 1e-10)
			x = tf.clip_by_value(x, 0.0, 1.0)

		if self.noise_range is not None:
			# Add random noise.
			noise_len = self.noise_range[1] - self.noise_range[0]
			noise_val = tf.cast(tf.random.uniform(tf.shape(x)), tf.float32)
			if self.noise_mask is not None:
				noise_val *= self.noise_mask

			if self.noise_replace and self.noise_mask is not None:
				x = x * (1.0-self.noise_mask) + \
					noise_val * noise_len + self.noise_range[0]
			else:
				x += noise_val * noise_len + self.noise_range[0]
			# Keep random noise or changes in dx from causing out-of-range values.
			x = tf.clip_by_value(x, 0.0, 1.0)
				
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
		for i in range(hsize):
			arr[:, self.img_size*i : self.img_size*(i+1), :] = x[..., i*3 : (i+1)*3]

		rgb_array = np.uint8(arr * 255.0)

		# Scale the first two dimensions of the image by the given scale.
		for dim in range(2):
				rgb_array = np.repeat(rgb_array, scale, dim)

		return PIL.Image.fromarray(rgb_array)
	
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

	def get_sum(self, x):
		return tf.reduce_sum(x)

	@tf.function
	def train_step(self, x0, xf, lifetime, lock_release=None, loss_f=None):
		if loss_f is None: loss_f = lambda x: loss_mean_square(x, xf)

		x = x0
		with tf.GradientTape() as g:
			for i in tf.range(lifetime):
				x = self.ca(x, lock_release is not None and i >= lock_release)
			loss = loss_f(x)
				
		grads = g.gradient(loss, self.ca.weights)
		grads = [g / (tf.norm(g) + 1.0e-8) for g in grads]
		self.trainer.apply_gradients(zip(grads, self.ca.weights))
		return x, loss

	def do_sample_run(self, x0, xf, lifetime, lock_release=None):
		# Run the CA for its lifetime with the current weights.
		x = x0()[None, ...]
				
		xs = []
		xs.append(x[0, ...])
		for i in range(lifetime):
			x = self.ca(x, lock_release is not None and i >= lock_release)
			xs.append(x[0, ...])

		return xs
	
	def show_sample_run(self, x0, xf, lifetime, lock_release=None):
		if xf:
			print("Target:")
			self.ca.display(xf())

		xs = self.do_sample_run(x0, xf, lifetime, lock_release)
		print("mass at t0:", self.ca.get_mass(x0()))
		print("mass at tf:", self.ca.get_mass(xs[-1]))
	
		print("Sample run:")
		self.ca.display_gif(xs)
		return xs

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
		max_plateau_len=None, lock_release=None, loss_f=None):
		if self.is_done(): return

		initial = loss = None
		start = time.time()
		elapsed_seconds = 0.0
		def show_elapsed_time():
			print("Time: ", elapsed_seconds, "seconds")

		best_loss = None
		plateau = 0

		num_steps = 0
		while True:
			if max_steps is not None:
				if num_steps >= max_steps: 
					print("Stopping due to max steps reached")
					show_elapsed_time()
					return
			if max_seconds is not None:
				if elapsed_seconds >= max_seconds: 
					print("Stopping due to time-out")
					show_elapsed_time()
					return
			if target_loss is not None:
				if self.loss_hist and self.loss_hist[-1] <= target_loss: 
					print("Stopping due to target loss reached")
					show_elapsed_time()
					return
			if max_plateau_len is not None:
				if plateau >= max_plateau_len:
					print("Stopping due to plateau")
					show_elapsed_time()
					return
					
			initial = np.repeat(x0()[None, ...], 1, 0)
			target = np.repeat(xf()[None, ...], 1, 0) if xf is not None else None
			_, loss = self.train_step(initial, target, lifetime, lock_release, loss_f)
			if best_loss is None or loss.numpy() < best_loss:
				best_loss = loss.numpy()
				plateau = 0
			else:
				plateau += 1

			self.loss_hist.append(loss.numpy())
			elapsed_seconds = time.time() - start
			num_steps += 1

			wandb.log(dict(loss=loss.numpy()), step=len(self.loss_hist))

			if self.is_done(): 
				print("Stopping due to zero loss")
				show_elapsed_time()
				return
					
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
	training = Training(ca=ca, learning_rate=learning_rate)
	return training


def sconf_zero_everywhere(ca: CellularAutomata):
	return ca.constfilled(0.0)

def sconf_image(filename: str):
	def f(ca: CellularAutomata):
		return ca.imagefilled(filename)
	return f

def sconf_center_black_dot(ca: CellularAutomata):
	return ca.pointfilled(ca.constfilled(1.0), point_value=0.0)

class Config(object):
	def __init__(self):
		self.size = 16
		self.num_channels = 3
		self.layer1_size = 64
		self.learning_rate = 1.0e-3
		self.training_seconds = 30
		self.num_sample_runs = 5
		self.edge_strategy = 'EdgeStrategy.MIRROR'
		self.initial_state = 'sconf_center_black_dot'
		self.target_state = 'sconf_image("lenna.png")'
		self.lifetime = 25

	def randomized(self):
		self.size = random.randrange(4,40)
		self.num_channels = random.randrange(1,5) * 3
		self.layer1_size = random.randrange(4,256)
		self.learning_rate = random.random() * 0.01 + 0.001
		self.lifetime = random.randrange(8,80)
		return self

def run_once(group: str, config: Config) -> None:
	wandb.init(project="neural-cellular-automata", group=group, config=vars(config))

	layer_counts = []
	if config.layer1_size: layer_counts.append(config.layer1_size)

	ca = CellularAutomata(img_size=config.size, channel_count=config.num_channels,
		layer_counts=layer_counts, perception_kernel=sobel_state_kernel())
	ca.edge_strategy = eval(config.edge_strategy)
	training = Training(ca=ca, learning_rate=config.learning_rate)

	x0 = eval(config.initial_state)(ca)
	xf = eval(config.target_state)(ca)
	x0_fn = lambda: x0
	xf_fn = lambda: xf

	interval_seconds = config.training_seconds / config.num_sample_runs

	for i in range(config.num_sample_runs):
		training.run(x0_fn, xf_fn, config.lifetime, max_seconds=interval_seconds)
		ca.model.save(os.path.join(wandb.run.dir, f"model_{i}.h5"))

		# Save a sample run:
		sample_run = training.do_sample_run(x0_fn, xf_fn, config.lifetime)
		gif_path = f"temp/sample_run_{i}.gif"
		with open(gif_path, 'wb') as gif:
			gif.write(ca.create_gif(sample_run))
		final_img = ca.to_image(sample_run[-1])
		wandb.log({
			f"final_state": wandb.Image(final_img),
			f"video": wandb.Video(gif_path)},
			step=len(training.loss_hist))

test_config = Config()
test_config.training_seconds = 10

def compare_learning_rates(group: str) -> None:
	config = Config()
	for x in np.linspace(0.0025, 0.0040, num=30):
		config.learning_rate = x
		run_once(group, config)