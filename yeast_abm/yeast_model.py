from typing import List
import time
import pandas as pd
import tensorflow as tf
import wandb
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
			tf.keras.layers.Dense(256, activation="relu"),
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
			rows.append([self.names[i], round(float(value), 3)])
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
		s0 = tf.constant(s0)
		with tf.GradientTape() as tape:
			snapshots = self.run_snapshots(s0, time_segments)
			loss = self.loss_snapshots(snapshots, targets)
		grads = tape.gradient(loss, self.nn.trainable_variables)
		self.optimizer.apply_gradients(zip(grads, self.nn.trainable_variables))
		return loss

class TrainingModel(object):
	def __init__(self, s0, time_segments, targets, network):
		self.s0 = s0
		self.time_segments = time_segments
		self.network = network
		self.targets = targets

	def display(self):
		snapshots = self.network.run_snapshots(self.s0, self.time_segments)
		for snapshot in snapshots:
			display(self.network.to_dataframe(snapshot))

	def train(self, start_idx: int, end_idx: int) -> float:
		s0 = self.s0
		if start_idx > 0:
			s0 = self.targets[start_idx - 1]

		loss = self.network.train(s0, self.targets[start_idx : end_idx],
			self.time_segments[start_idx : end_idx])
		return loss

def main():
	config = dict(
		num_stages=9, num_trials=10, train_type="Graduated Segments w/ Segment Isolation"
	)
	wandb.init(project="neural-cellular-automata", group="yeast_model_1", config=config)

	times = []
	for i in range(config['num_trials']):
		network = ProteinNetwork([
			"SK", "Cdc2/Cdc13", "Ste9", "Rum1", "Slp1", "Cdc2/Cdc13*", "Wee1Mik1", "Cdc25", "PP"])
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
		assert(len(targets) == config['num_stages'])
		time_segments = [5 for _ in targets]

		model = TrainingModel(s0, time_segments, targets, network)

		start = time.time()
		for i in range(len(time_segments)):
			print("Doing", i+1, "segments")
			target_loss = 0.1
			loss = 9999.9
			while loss > target_loss:
				# Isolated segment:
				# for _ in range(10):
				# 	model.train(i, i+1)

				# Graduated segments:
				for _ in range(50):
					loss = model.train(0, i+1)
				print("Loss=", loss.numpy())

		t = time.time() - start
		print(t, "seconds to train graduated segments")
		times.append(t)
	
	print("Total training time: ", sum(times))
	wandb.run.summary['mean_training_time'] = sum(times) / len(times)