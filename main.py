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

def build_and_train(group: str, config: Config, ca_modifier_fn=None,
	use_trained_ca: TrainedCa=None) -> TrainedCa:
	wandb.init(project="neural-cellular-automata", group=group, config=vars(config))

	perception_kernel = None
	if config.perception_kernel_size == 0 or config.perceive_layer_size == 0:
		perception_kernel = kernel_sobel()
	ca = CellularAutomata(config, perception_kernel)
	if use_trained_ca is not None:
		ca.copy_weights_from(use_trained_ca.ca)

	if ca_modifier_fn: ca_modifier_fn(ca)

	training = Training(ca=ca, config=config)

	x0 = eval(config.initial_state)(ca)
	xf = eval(config.target_state)(ca)
	#loss_fn = eval(config.loss_fn)(xf)
	x0_fn = lambda: x0
	xf_fn = lambda: xf

	interval_seconds = config.training_seconds / config.num_sample_runs

	start = time.time()

	def save_sample_run(i: int):
		sample_run = training.do_sample_run(x0_fn, config.lifetime)
		gif_path = f"temp/sample_run_{i}.gif"
		with open(gif_path, 'wb') as gif:
			gif.write(ca.create_gif(sample_run))
		final_img = ca.to_image(sample_run[-1])
		wandb.log({
			f"final_state": wandb.Image(final_img),
			f"video": wandb.Video(gif_path)},
			step=len(training.loss_hist))

	save_sample_run(0)

	for i in range(config.num_sample_runs):
		if config.use_growing_square:
			lifetime = (config.lifetime // config.num_sample_runs) * (i+1)
			target_size = (config.size // config.num_sample_runs) * (i+1)
		else:
			lifetime = config.lifetime
			target_size = config.size
		a = config.size // 2 - target_size // 2
		b = config.size // 2 + target_size // 2
		print("Lifetime: ", lifetime)
		print("Target size: ", target_size)

		def loss_fn(x):
			x = x[:, a:b, a:b, :config.target_channels]
			f = xf[None, a:b, a:b, :config.target_channels]
			lx = CellularAutomata.laplacian(x)
			lf = CellularAutomata.laplacian(f)
			laplace_err = tf.reduce_mean(tf.square(lx - lf))
			mse = tf.reduce_mean(tf.square(x - f))
			return mse + laplace_err

		training.run(x0_fn, xf_fn, lifetime, loss_fn, max_seconds=interval_seconds)

		ca.model.save(os.path.join(wandb.run.dir, f"model_{i}.h5"))
		if ca.perception_model is not None:
			ca.perception_model.save(os.path.join(wandb.run.dir, f"perceive_model_{i}.h5"))

		save_sample_run(i+1)
		
		best_so_far = min(training.loss_hist)
		print("Best loss: ", best_so_far)

	elapsed_total = time.time() - start
	print("Total elapsed time:", elapsed_total, "seconds")

	return TrainedCa(ca, training)

def main():
	config = Config()
	config.layer1_size = 256
	config.num_channels = 15
	config.target_channels = 3
	config.training_seconds = 60
	#config.target_loss = 0.01
	config.num_sample_runs = 3
	config.lifetime = 50
	config.size = 25
	config.initial_state = 'sconf_center_black_dot'
	config.edge_strategy = 'EdgeStrategy.TF_SAME'
	config.use_growing_square = False

	# Sanity check that copying models works:

	config.target_state = 'sconf_image("lenna.png")'
	res1 = build_and_train("multi_image", config)

	config.target_state = 'sconf_image("nautilus.png")'
	res2 = build_and_train("multi_image", config, use_trained_ca=res1)

	config.target_state = 'sconf_image("lenna.png")'
	res3 = build_and_train("multi_image", config, use_trained_ca=res1)

def comparing_stacked_vs_separate():
	""" In this experiment we check if learning multiple images in parallel
	takes less time than learning the images separately. """

	config = Config()
	config.layer1_size = 256
	config.num_channels = 15
	config.target_channels = 3
	config.training_seconds = 999
	config.target_loss = 0.01
	config.num_sample_runs = 10
	config.lifetime = 64
	config.size = 32
	config.initial_state = 'sconf_center_black_dot'
	config.edge_strategy = 'EdgeStrategy.TF_SAME'

	for _ in range(1):
		config.target_state = 'sconf_image("lenna.png")'
		build_and_train("stacked_training", config)
	for _ in range(1):
		config.target_state = 'sconf_image("nautilus.png")'
		build_and_train("stacked_training", config)
	for _ in range(1):
		config.target_channels = 6
		config.target_state = 'sconf_imagestack("lenna.png", "nautilus.png")'
		build_and_train("stacked_training", config)