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

	def __init__(self, node_names: List[str], num_channels: int):
		self.names = node_names
		self.num_channels = num_channels

		self.nn = tf.keras.models.Sequential([
			tf.keras.layers.Input(shape=(len(node_names)*2-1,num_channels)),
			tf.keras.layers.Conv1D(256, kernel_size=len(node_names),
				activation="relu", padding="VALID"),
			tf.keras.layers.Conv1D(num_channels, kernel_size=1, activation=None,
				kernel_initializer=tf.zeros_initializer())
			# # Input layer that takes in each protein:
			# tf.keras.layers.Input(shape=(len(node_names),)),
			# # Hidden layer that simulates interactions between proteins:
			# tf.keras.layers.Dense(256, activation="relu"),
			# # Output layer that produces change in each protein's activation amount:
			# tf.keras.layers.Dense(1, activation=None),
		])
		self.nn.build()
		self.nn.summary()

		self.optimizer = tf.keras.optimizers.Adam()

	def set_activation(self, x, protein: str, activation: float):
		idx = self.names.index(protein)
		x[idx] = activation

	def get_activation(self, x, protein: str):
		idx = self.names.index(protein)
		return x[idx].numpy()

	def zeros(self):
		return [0.0 for protein in self.names]

	@tf.function
	def tick_once(self, x):
		dx = self.nn(tf.reshape(tf.tile(x, [2,1])[:-1], [1,2*len(self.names)-1,self.num_channels]))
		dx = tf.reshape(dx[0][:len(self.names)], x.shape)
		return x + dx

	def run_for_ticks(self, s0, num_ticks):
		s = s0
		for _ in range(num_ticks):
			s = self.tick_once(s)
		return s

	@staticmethod
	def loss(s, target):
		""" Calculate the mean squared error that will be used 
		as the training loss. """
		return tf.reduce_mean(tf.square(s[:, 0] - target))

	@staticmethod
	def loss_snapshots(s_snapshots, target_snapshots):
		loss = 0
		for s, target in zip(s_snapshots, target_snapshots):
			loss += ProteinNetwork.loss(s, target)
		return loss

	def to_dataframe(self, s):
		rows = []
		for i, value in enumerate(s):
			rows.append([self.names[i], round(float(value[0]), 3)])
		df = pd.DataFrame(rows, columns=["protein", "activation"])
		return df

	def run_snapshots(self, s0, time_segments=[]):
		""" Pass in a list of time step amounts to run 
		the model for each of those segments, producing the output
		at each snapshot of time at the end of each segment. """
		res = []
		s0 = tf.reshape(
			tf.stack([s0 for _ in range(self.num_channels)]),
			[len(self.names), self.num_channels])
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

class TrainingModel(object):
	def __init__(self, time_segments, targets, network):
		self.time_segments = time_segments
		self.network = network
		self.targets = targets

	def display(self):
		snapshots = self.network.run_snapshots(self.targets[0], self.time_segments)
		# self.network.to_dataframe(self.targets[0])
		for i, snapshot in enumerate(snapshots):
			print("Snapshot", i)
			display(self.network.to_dataframe(snapshot))

	def train(self, end_idx: int) -> float:
		loss = self.network.train(self.targets[0], self.targets[1 : end_idx+1],
			self.time_segments[0 : end_idx])
		return loss
	
	def sample_an_interaction(self, protein1: str, protein2: str, num_steps: int):
		""" Create an interaction table between two proteins, testing 
		the two in isolation in the four different activation configurations 
		they could be in. """

		rows = []
		for p1_activation in [0., 1.]:
			for p2_activation in [0., 1.]:
				s0 = self.network.zeros()
				self.network.set_activation(s0, protein1, p1_activation)
				self.network.set_activation(s0, protein2, p2_activation)
				result = self.network.run_snapshots(s0, [num_steps])[0]
				p1_result = self.network.get_activation(result, protein1)
				p2_result = self.network.get_activation(result, protein1)
				rows.append([p1_activation, p2_activation, p1_result, p2_result])

		df = pd.DataFrame(rows, columns=[protein1, protein2, 
			protein1 + " (t+1)", protein2 + " (t+1)"])
		display(df)

def run_experiment(config):
	run = wandb.init(project="neural-cellular-automata", group="final_yeast_abm", config=config)

	network = ProteinNetwork([
		"SK", "Cdc2/Cdc13", "Ste9", "Rum1", "Slp1", "Cdc2/Cdc13*", "Wee1Mik1", "Cdc25", "PP"],
		num_channels=config["num_channels"])
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
	
	time_segments = [
		5, # S
		5, # G2
		7, # G2
		7, # G2
		7, # G2
		7, # M
		3, # M
		3, # G1
	]

	model = TrainingModel(time_segments, targets, network)
	timeout = 1000

	start = time.time()

	def do_training():
		target_loss = config['target_loss']
		if config['gradual']:
			for i in range(len(time_segments)):
				print("Doing", i+1, "segment(s)")
				loss = 9999.9
				while loss > target_loss:
					if time.time() - start > timeout:
						print("Stopping due to time out")
						return
					# Graduated segments:
					loss = model.train(i+1)
					wandb.log({"loss": loss})

		else:
			print("Doing", len(time_segments), "segments")
			loss = 9999.9
			while loss > target_loss:
				if time.time() - start > timeout:
					print("Stopping due to time out")
					return
				loss = model.train(len(time_segments))
				wandb.log({"loss": loss})
	do_training()

	t = time.time() - start
	print(t, "seconds to train")
	wandb.run.summary["total_seconds"] = t

	model.display()
	run.finish()

	# LEGEND:
	# -> inhibits
	# <-> inhibits both ways
	# => activates
	# <=> activates both ways
	# x self-inhibits

	# print("SK x-> Ste9")
	# model.sample_an_interaction("SK", "Ste9", num_steps=5)
	# print("SK x-> Rum1")
	# model.sample_an_interaction("SK", "Rum1", num_steps=5)
	# print("Ste9 <-> Cdc2/Cdc13")
	# model.sample_an_interaction("Ste9", "Cdc2/Cdc13", num_steps=5)
	# print("Rum1 <-> Cdc2/Cdc13")
	# model.sample_an_interaction("Rum1", "Cdc2/Cdc13", num_steps=5)

def main():
	config = dict(
		target_loss=0.01,
	)

	while True:
		for n in [16,8,4,2]:
			config['num_channels'] = n
			config['gradual'] = True
			run_experiment(config)
			config['gradual'] = False
			run_experiment(config)

if __name__ == "__main__":
	main()