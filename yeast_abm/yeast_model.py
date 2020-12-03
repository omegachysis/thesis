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

	def to_dataframe(self, s):
		rows = []
		for i, value in enumerate(s):
			rows.append([self.names[i], float(value)])
		df = pd.DataFrame(rows, columns=["protein", "activation"])
		return df

	def train_step(self, s0, target, num_ticks):
		""" Return the loss from this training step. """
		with tf.GradientTape() as tape:
			sf = self.run_for_ticks(s0, num_ticks)
			loss = self.loss(sf, target)
		grads = tape.gradient(loss, self.nn.trainable_variables)
		self.optimizer.apply_gradients(zip(grads, self.nn.trainable_variables))
		return loss

def main():
	network = ProteinNetwork(["test1", "test2", "test3", "test4"])
	s0 = network.zeros()
	for i in range(100):
		print("Train step", i)
		loss = network.train_step(s0, target=network.ones(), num_ticks=10)
		print("Loss=", loss.numpy())
	s = s0
	for i in range(10):
		display(network.to_dataframe(s))
		s = network.tick_once(s)
	display(network.to_dataframe(s))