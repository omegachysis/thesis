from config import Config
from urllib.parse import non_hierarchical
import tensorflow as tf
import numpy as np
import time
import wandb
from matplotlib import pyplot as plt

class Training(object):
	def __init__(self, ca, config):
		self.ca = ca
		self.loss_hist = []
		self.trainer = tf.keras.optimizers.Adam(config.learning_rate,
			epsilon=config.epsilon)
		self.target_loss = config.target_loss

	@tf.function
	def train_step(self, x0, xf, lifetime, loss_fn, target_channels, lock_release: int=None):
		x = x0
		with tf.GradientTape() as g:
			for i in tf.range(lifetime):
				#x = self.ca(x, lock_release is not None and i >= lock_release)
				x = self.ca(x)
			losses = []
			for channel in range(target_channels):
				losses.append(loss_fn(x, channel))
			loss = tf.reduce_mean(losses)
				
		grads = g.gradient(loss, self.ca.model.trainable_variables)
		grads = [g / (tf.norm(g) + 1.0e-8) for g in grads]
		self.trainer.apply_gradients(zip(grads, self.ca.model.trainable_variables))
		return x, losses

	def do_sample_run(self, x0, lifetime):
		# Run the CA for its lifetime with the current weights.
		x = x0()[None, ...]
				
		xs = []
		xs.append(x[0, ...])
		for i in range(lifetime):
			#x = self.ca(x, lock_release is not None and i >= lock_release)
			x = self.ca(x)
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
		return self.loss_hist and self.loss_hist[-1] <= self.target_loss
	
	def run(self, x0, xf, lifetime: int, loss_fn, target_channels, 
	max_seconds=None, max_plateau_len=None):
		initial = loss = None
		start = time.time()
		elapsed_seconds = 0.0
		def show_elapsed_time():
			print("Time: ", elapsed_seconds, "seconds")

		best_loss = None
		plateau = 0

		num_steps = 0
		while True:
			if max_seconds is not None:
				if elapsed_seconds >= max_seconds: 
					print("Stopping due to time-out")
					show_elapsed_time()
					return
			if max_plateau_len is not None:
				if plateau >= max_plateau_len:
					print("Stopping due to plateau")
					show_elapsed_time()
					return
					
			initial = np.repeat(x0()[None, ...], 1, 0)
			target = np.repeat(xf()[None, ...], 1, 0) if xf is not None else None

			# Run training step:
			_, losses = self.train_step(initial, target, lifetime, loss_fn, target_channels)
			loss = tf.reduce_mean(losses)

			# Update best loss and increment plateau:
			if best_loss is None or loss.numpy() < best_loss:
				best_loss = loss.numpy()
				plateau = 0
			else:
				plateau += 1

			self.loss_hist.append(loss.numpy())
			elapsed_seconds = time.time() - start
			num_steps += 1

			log_data = {"loss" : loss.numpy(), "step" : len(self.loss_hist)}
			for i in range(target_channels // 3):
				log_data[f"loss{i}"] = tf.reduce_mean([losses[j] for j in range(i*3,i*3+3)])
			wandb.log(log_data)

			if self.is_done(): 
				print("Stopping due to target loss")
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