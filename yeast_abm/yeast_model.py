from typing import List
import pandas as pd
import tensorflow as tf
from IPython.display import display

class ProteinNetwork(object):
	""" Encodes a complete graph 
	structure where each graph node is a protein 
	and the edges represent interactions between proteins. """

	def __init__(self, node_names: List[str]):
		self.names = node_names

		self.nn = tf.keras.models.Sequential([
			# Input layer that takes in each protein:
			tf.keras.layers.Input(shape=(len(node_names),)),
			# Hidden layer that simulates interactions between proteins:
			tf.keras.layers.Dense(64, activation="relu"),
			# Output layer that produces change in each protein's activation amount:
			tf.keras.layers.Dense(len(node_names), activation=None),
		])
		self.nn.build()
		self.nn.summary()

		self.optimizer = tf.keras.optimizers.Adam()

	def ones(self):
		return tf.ones(shape=(len(self.names,)))
	def zeros(self):
		return tf.zeros(shape=(len(self.names,)))

	@tf.function
	def tick_once(self, x):
		return x + self.nn(x[None, ...])[0]

	def run_for_ticks(self, s0, num_ticks):
		s = s0
		for _ in range(num_ticks):
			s = self.tick_once(s)
		return s

	@staticmethod
	def loss(s, target):
		""" Calculate the mean squared error that will be used 
		as the training loss. """
		return tf.reduce_mean(tf.square(s - target))

	@staticmethod
	def loss_snapshots(s_snapshots, target_snapshots):
		loss = 0
		for s, target in zip(s_snapshots, target_snapshots):
			loss += ProteinNetwork.loss(s, target)
		return loss

	def to_dataframe(self, s):
		rows = []
		for i, value in enumerate(s):
			rows.append([self.names[i], float(value)])
		df = pd.DataFrame(rows, columns=["protein", "activation"])
		return df

	def run_snapshots(self, s0, time_segments=[]):
		""" Pass in a list of time step amounts to run 
		the model for each of those segments, producing the output
		at each snapshot of time at the end of each segment. """
		res = []
		s = s0
		for num_ticks in time_segments:
			s = self.run_for_ticks(s, num_ticks)
			res.append(s)
		return res

	def train(self, s0, targets=[], time_segments=[]):
		""" Return the loss from this training step. """
		assert len(targets) == len(time_segments)
		with tf.GradientTape() as tape:
			snapshots = self.run_snapshots(s0, time_segments)
			loss = self.loss_snapshots(snapshots, targets)
		grads = tape.gradient(loss, self.nn.trainable_variables)
		self.optimizer.apply_gradients(zip(grads, self.nn.trainable_variables))
		return loss

def main():
	network = ProteinNetwork([
		"SK", "Cdc2/Cdc13", "Ste9", "Rum1", "Slp1", "Cdc2/Cdc13*", "Wee1Mik1", "Cdc25", "PP"])
	time_segments = [10 for _ in range(9)]
	s0 = network.zeros()
	targets = [
		[1., 0.,    1., 1., 0., 0.,   1.,    0., 0.], # G1
		[0., 0.,    0., 0., 0., 0.,   1.,    0., 0.], # S
		[0., 1.,    0., 0., 0., 0.,   1.,    0., 0.],	# G2
		[0., 1.,    0., 0., 0., 0.,   0.,    1., 0.],	# G2
		[0., 1.,    0., 0., 0., 1.,   0.,    1., 0.],	# G2
		[0., 1.,    0., 0., 1., 1.,   0.,    1., 0.],	# G2
		[0., 0.,    0., 0., 1., 0.,   0.,    1., 1.],	# M
		[0., 0.,    1., 1., 0., 0.,   1.,    0., 1.],	# M
		[0., 0.,    1., 1., 0., 0.,   1.,    0., 0.],	# G1
	]

	for i in range(len(time_segments)):
		print("Doing", i+1, "segments")
		for _ in range(10):
			loss = 0.0
			for _ in range(100):
				loss = network.train(s0, targets[:i+1], time_segments[:i+1])
			print("Loss=", loss.numpy())

		snapshots = network.run_snapshots(s0, time_segments[:i+1])
		for snapshot in snapshots[:i+1]:
			display(network.to_dataframe(snapshot))


