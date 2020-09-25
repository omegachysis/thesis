import os
import wandb

from config import *
from ca import *
from training import *

def run_once(group: str, config: Config) -> None:
	wandb.init(project="neural-cellular-automata", group=group, config=vars(config))

	layer_counts = []
	if config.layer1_size: layer_counts.append(config.layer1_size)
	if config.layer2_size: layer_counts.append(config.layer2_size)

	ca = CellularAutomata(img_size=config.size, channel_count=config.num_channels,
		layer_counts=layer_counts, perception_kernel=kernel_sobel())
	ca.edge_strategy = eval(config.edge_strategy)
	training = Training(ca=ca, learning_rate=config.learning_rate)

	x0 = eval(config.initial_state)(ca)
	xf = eval(config.target_state)(ca)
	loss_fn = eval(config.loss_fn)(xf)
	x0_fn = lambda: x0
	xf_fn = lambda: xf

	interval_seconds = config.training_seconds / config.num_sample_runs

	best_loss = 999
	for i in range(config.num_sample_runs):
		training.run(x0_fn, xf_fn, config.lifetime, loss_fn, max_seconds=interval_seconds)

		ca.model.save(os.path.join(wandb.run.dir, f"model_{i}.h5"))

		# Save a sample run:
		sample_run = training.do_sample_run(x0_fn, xf_fn, config.lifetime)
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
		if best_so_far < best_loss:
			best_loss = best_so_far
		else:
			print("Stopping early due to loss plateau...")
			break

def main():
	config = Config()
	run_once("refactored_1", config)