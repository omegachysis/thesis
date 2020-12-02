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
		self.values = tf.zeros(shape=(len(node_names),))

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

	def to_dataframe(self):
		rows = []
		for i, value in enumerate(self.values):
			rows.append([self.names[i], float(value)])
		df = pd.DataFrame(rows, columns=["protein", "activation"])
		return df

def main():
	network = ProteinNetwork(["test1", "test2", "test3", "test4"])
	display(network.to_dataframe())