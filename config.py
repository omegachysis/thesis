import tensorflow as tf
import numpy as np
import random
from ca import CellularAutomata

# KERNELS ----------------------------------------

def kernel_sobel():
	# Create a Sobel filter:
	identity = np.float32([0, 1, 0])
	identity = np.outer(identity, identity)
	dx = np.outer(np.float32([1, 2, 1]), np.float32([-1, 0, 1])) / 8.0
	dy = dx.T
	return tf.stack([identity, dx - dy, dx + dy], axis=-1)

def kernel_neighbors():
	basis = []
	for x in range(3):
		for y in range(3):
			tensor = np.zeros((3, 3), dtype=np.float32)
			tensor[x, y] = 1
			basis.append(tensor)
	return tf.stack(basis, axis=-1)

# STATES ----------------------------------------

def sconf_zero_everywhere(ca: CellularAutomata):
	return ca.constfilled(0.0)

def sconf_image(filename: str):
	def f(ca: CellularAutomata):
		return ca.imagefilled("images/" + filename)
	return f

def sconf_center_black_dot(ca: CellularAutomata):
	return ca.pointfilled(ca.constfilled(1.0), point_value=0.0)

# LOSS FUNCTIONS ----------------------------------------------------

def loss_mse(target):
	def f(x):
		return tf.reduce_mean(tf.square(x[...,:3] - target[...,:3]))
	return f

# --------------------------------------------------------------------

class Config(object):
	def __init__(self):
		self.size = 16
		self.num_channels = 3
		self.layer1_size = 64
		self.layer2_size = 0
		self.learning_rate = 3.0e-3
		self.training_seconds = 30
		self.num_sample_runs = 5
		self.edge_strategy = 'EdgeStrategy.MIRROR'
		self.initial_state = 'sconf_center_black_dot'
		self.target_state = 'sconf_image("lenna.png")'
		self.loss_fn = 'loss_mse'
		self.lifetime = 25

	def randomized(self):
		self.size = random.randrange(20,60)
		self.num_channels = random.randrange(1,3) * 3
		self.layer1_size = random.randrange(5,64)
		self.layer2_size = random.randrange(5,64)
		self.learning_rate = random.random() * 0.001 + 0.001
		self.lifetime = self.size + random.randrange(self.size)
		return self