import tensorflow as tf
import numpy as np
import wandb
import os
import random

def main():
	config = {
		"neuron_count": 64,
	}

	for neuron_count in [16, 32, 48, 64]:
		config["neuron_count"] = neuron_count
		
		wandb.init(project="test", config=config)

		model = tf.keras.models.Sequential([
			tf.keras.layers.Input(shape=(1,)),
			tf.keras.layers.Dense(config["neuron_count"], activation='relu'),
			tf.keras.layers.Dense(1),
		])

		x_train = np.array([random.randrange(100) for _ in range(1000)])
		y_train = np.array([x for x in x_train])

		model.compile(optimizer="adam",
			loss=tf.keras.losses.MeanSquaredError())

		for i in range(10):
			history = model.fit(x_train, y_train)
			loss = history.history["loss"][-1]
			wandb.log({"loss": loss})

		model.save(os.path.join(wandb.run.dir, "model.h5"))