import os
import wandb
from typing import Tuple

from config import *
from ca import *
from training import *

class TrainedCa(object):
	def __init__(self, ca: CellularAutomata, training: Training):
		self.ca = ca
		self.training = training

def build_and_train(group: str, config: Config, ca_modifier_fn=None) -> TrainedCa:
	wandb.init(project="neural-cellular-automata", group=group, config=vars(config))

	layer_counts = []
	if config.layer1_size: layer_counts.append(config.layer1_size)
	if config.layer2_size: layer_counts.append(config.layer2_size)

	ca = CellularAutomata(img_size=config.size, channel_count=config.num_channels,
		layer_counts=layer_counts, perception_kernel=kernel_sobel(),
		num_subnetworks=config.num_subnetworks, combiner_layer_size=config.combiner_layer_size)
	ca.edge_strategy = eval(config.edge_strategy)
	ca.clamp_values = config.clamp_values

	if ca_modifier_fn: ca_modifier_fn(ca)

	training = Training(ca=ca, config=config)

	x0 = eval(config.initial_state)(ca)
	xf = eval(config.target_state)(ca)
	#loss_fn = eval(config.loss_fn)(xf)
	x0_fn = lambda: x0
	xf_fn = lambda: xf

	interval_seconds = config.training_seconds / config.num_sample_runs

	start = time.time()

	for i in range(config.num_sample_runs):
		lifetime = (config.lifetime // config.num_sample_runs) * (i+1)
		target_size = (config.size // config.num_sample_runs) * (i+1)
		a = config.size // 2 - target_size // 2
		b = config.size // 2 + target_size // 2
		print("Lifetime: ", lifetime)
		print("Target size: ", target_size)

		def loss_fn(x):
			x = x[:, a:b, a:b, :3]
			f = xf[None, a:b, a:b, :3]
			lx = CellularAutomata.laplacian(x)
			lf = CellularAutomata.laplacian(f)
			laplace_err = tf.reduce_mean(tf.square(lx - lf))
			mse = tf.reduce_mean(tf.square(x - f))
			return laplace_err + mse

		training.run(x0_fn, xf_fn, lifetime, loss_fn, max_seconds=interval_seconds)

		ca.model.save(os.path.join(wandb.run.dir, f"model_{i}.h5"))

		# Save a sample run:
		sample_run = training.do_sample_run(x0_fn, lifetime)
		gif_path = f"temp/sample_run_{i}.gif"
		with open(gif_path, 'wb') as gif:
			gif.write(ca.create_gif(sample_run))
		final_img = ca.to_image(sample_run[-1])
		wandb.log({
			f"final_state": wandb.Image(final_img),
			f"video": wandb.Video(gif_path)},
			step=len(training.loss_hist))
		
		best_so_far = min(training.loss_hist)
		print("Best loss: ", best_so_far)

	elapsed_total = time.time() - start
	print("Total elapsed time:", elapsed_total, "seconds")

	return TrainedCa(ca, training)

def main():
	num_steps = 4

	config = Config()
	config.layer1_size = 256
	config.training_seconds = num_steps*999
	config.target_loss = 0.05
	config.num_sample_runs = num_steps
	config.lifetime = 60
	config.size = 16
	config.initial_state = 'sconf_center_black_dot'
	config.target_state = 'sconf_image("nautilus.png")'
	config.edge_strategy = 'EdgeStrategy.MIRROR'
	trained_ca = build_and_train("trainable_filter", config)