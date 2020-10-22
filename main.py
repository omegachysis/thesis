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
	loss_fn = eval(config.loss_fn)(xf)
	x0_fn = lambda: x0
	xf_fn = lambda: xf

	interval_seconds = config.training_seconds / config.num_sample_runs

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

		if training.is_done():
			print("Target loss reached, ending...")
			break

	return TrainedCa(ca, training)

def main():
	config = Config()
	config.num_subnetworks = 1
	config.layer1_size = 128
	config.training_seconds = 10
	config.num_sample_runs = 1
	config.size = 16
	config.target_state = 'sconf_image("lenna.png")'
	trained_ca = build_and_train("net_transfer", config)
	trained_ca.ca.model.save_weights("temp/saved_weights", overwrite=True)

	# print("Trained model layers:")
	# for layer in trained_ca.ca.model.layers:
	# 	print(layer, layer.get_weights())

	# Transfer the trained network into a larger one with two subnetworks.
	config = Config()
	config.num_subnetworks = 2
	config.combiner_layer_size = 128
	config.layer1_size = 128
	config.training_seconds = 10
	config.num_sample_runs = 1
	config.size = 16
	config.target_state = 'sconf_image("lenna.png")'
	config.subnetworks_description = "first subnet: trained lenna; second subnet: nothing"
	def inject_ca(big_ca: CellularAutomata):
		big_ca.inject_into_submodel(submodel_idx=0, saved_model_path="temp/saved_weights")
	trained_ca2 = build_and_train("net_transfer", config, ca_modifier_fn=inject_ca)

	# print("Post final training model layers:")	
	# for layer in trained_ca2.ca.model.layers:
	# 	print(layer, layer.get_weights())