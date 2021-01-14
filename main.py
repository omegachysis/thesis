import os
import wandb
import glob
from typing import Tuple

from config import *
from ca import *
from training import *

def build_and_train(group: str, config: Config):
	wandb.init(project="neural-cellular-automata", group=group, config=vars(config))

	ca = CellularAutomata(config, kernel_sobel())
	training = Training(ca=ca, config=config)

	x0 = eval(config.initial_state)(ca)
	xf = eval(config.target_state)(ca)
	x0_fn = lambda: x0
	xf_fn = lambda: xf

	print("Target state:")
	ca.display(xf)

	window_size = config.growing_jump
	if window_size <= 0:
		window_size = config.size

	start = time.time()
	
	while True:
		print("Window size: ", window_size)
		lifetime = window_size

		a = config.size // 2 - window_size // 2
		b = config.size // 2 + window_size // 2
		a = max(0, a)
		b = min(config.size - 1, b)

		def loss_fn(x, channel):
			x = x[:, a:b, a:b, channel:channel+1]
			f = xf[None, a:b, a:b, channel:channel+1]
			lx = CellularAutomata.laplacian(x)
			lf = CellularAutomata.laplacian(f)
			# laplace_err = tf.reduce_mean(tf.square(lx - lf))
			rmse = tf.sqrt(tf.reduce_mean(tf.square(x - f)))
			return rmse

		training.run(x0_fn, xf_fn, lifetime, loss_fn, config.target_channels)

		if window_size >= config.size:
			break
		window_size += config.growing_jump

	elapsed_total = time.time() - start
	print("Total elapsed time:", elapsed_total, "seconds")
	wandb.run.summary["total_seconds"] = elapsed_total

	sample_run = training.do_sample_run(x0_fn, config.lifetime)
	gif_path = f"temp/sample_run.gif"
	with open(gif_path, 'wb') as gif:
		gif.write(ca.create_gif(sample_run))
	final_img = ca.to_image(sample_run[-1])
	wandb.log({
		f""
		f"final_state": wandb.Image(final_img),
		f"video": wandb.Video(gif_path)},
		step=len(training.loss_hist))
	return sample_run[-1]

def final_plain():
	""" In this experiment we just run the standard training algorithm 
	on all the final test images from the image database. """

	config = Config()
	config.layer1_size = 256
	config.num_channels = 18
	config.target_channels = 3
	config.target_loss = 0.01
	config.lifetime = 32
	config.size = 32
	config.initial_state = 'sconf_center_black_dot'
	config.edge_strategy = 'EdgeStrategy.TF_SAME'
	config.growing_jump = 0

	for path in glob.glob("images/final/*.png"):
		img_name = os.path.basename(path)
		config.target_state = f'sconf_image("final/{img_name}")'
		build_and_train("final_plain", config)

def final_center_growing():
	""" In this experiment we compare various center growing squares """

	config = Config()
	config.layer1_size = 256
	config.num_channels = 18
	config.target_channels = 3
	config.target_loss = 0.01
	config.lifetime = 32
	config.size = 32
	config.initial_state = 'sconf_center_black_dot'
	config.edge_strategy = 'EdgeStrategy.TF_SAME'

	for jump in range(1,6):
		config.growing_jump = jump
		for path in glob.glob("images/final/*.png"):
			img_name = os.path.basename(path)
			config.target_state = f'sconf_image("final/{img_name}")'
			build_and_train("final_center_growing", config)

def main():
	final_plain()