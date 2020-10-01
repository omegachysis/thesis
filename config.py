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

def sconf_one_everywhere(ca: CellularAutomata):
	return ca.constfilled(1.0)

def sconf_image(filename: str):
	def f(ca: CellularAutomata):
		return ca.imagefilled("images/" + filename)
	return f

def sconf_center_black_dot(ca: CellularAutomata):
	return ca.pointfilled(ca.constfilled(1.0), point_value=0.0)

# LOSS FUNCTIONS ----------------------------------------------------

def loss_mse(target):
	def f(x):
		a = x[..., :3] #tf.clip_by_value(x[...,:3], 0., 1.)
		b = target[..., :3] #tf.clip_by_value(target[...,:3], 0., 1.)
		return tf.reduce_mean(tf.square(a - b))
	return f

def loss_laplacian(target):
	target = target[None, ...]
	def f(x):
		lx = CellularAutomata.laplacian(x)
		lt = CellularAutomata.laplacian(target)
		return tf.reduce_mean(tf.square(lx[...,:3] - lt[...,:3]))
	return f

def loss_combined(loss1, loss2):
	def res(target):
		l1 = loss1(target)
		l2 = loss2(target)
		def f(x): return l1(x) + l2(x)
		return f
	return res

# --------------------------------------------------------------------

class Config(object):
	def __init__(self):
		self.size = 64
		self.num_channels = 15
		self.layer1_size = 256
		self.layer2_size = 0
		self.learning_rate = 3.0e-3
		self.epsilon = 1.0e-6
		self.training_seconds = 60 * 4
		self.num_sample_runs = 8
		self.edge_strategy = 'EdgeStrategy.ZEROS'
		self.initial_state = 'sconf_center_black_dot'
		self.target_state = 'sconf_image("lenna.png")'
		self.loss_fn = 'loss_mse'
		self.lifetime = 64
		self.clamp_values = False