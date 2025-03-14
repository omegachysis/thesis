import os
import wandb
import glob
from typing import Tuple

from config import *
from ca import *
from training import *

def build_and_train(group: str, config: Config):
	run = wandb.init(project="neural-cellular-automata", group=group, config=vars(config))

	ca = CellularAutomata(config, eval(config.kernel_set))
	training = Training(ca=ca, config=config)

	x0 = eval(config.initial_state)(ca)
	xf = eval(config.target_state)(ca)
	x0_fn = lambda: x0
	xf_fn = lambda: xf

	print("Target state:")
	ca.display(xf)

	# print("Kernel set:")
	# print(ca.perception_kernel)

	window_size = 3
	if config.growing_jump <= 0:
		window_size = config.size

	start = time.time()
	
	while True:
		print("Window size: ", window_size)
		lifetime = window_size + 10
		print("Lifetime: ", lifetime)

		def rmse_loss(x, channel):
			x = x[:, :, :, channel:channel+1]
			f = xf[None, :, :, channel:channel+1]
			return tf.sqrt(tf.reduce_mean(tf.square(x - f)))

		def rmse_laplace_loss(x, channel):
			x = x[:, :, :, channel:channel+1]
			f = xf[None, :, :, channel:channel+1]
			rmse = tf.sqrt(tf.reduce_mean(tf.square(x - f)))
			lx = CellularAutomata.laplacian(x)
			lf = CellularAutomata.laplacian(f)
			laplace_err = tf.reduce_mean(tf.square(lx - lf))
			return rmse + laplace_err

		loss_fn = rmse_laplace_loss if config.laplace_loss else rmse_loss

		training.run(x0_fn, xf_fn, lifetime, loss_fn, config.target_channels,
			max_seconds=1000)

		if window_size >= config.size:
			break
		window_size += config.growing_jump

	elapsed_total = time.time() - start
	print("Total elapsed time:", elapsed_total, "seconds")
	wandb.run.summary["total_seconds"] = elapsed_total

	sample_run = training.do_sample_run(x0_fn, config.size + 10)
	gif_path = f"temp/sample_run.gif"
	with open(gif_path, 'wb') as gif:
		gif.write(ca.create_gif(sample_run))
	final_img = ca.to_image(sample_run[-1])
	wandb.log({
		f""
		f"final_state": wandb.Image(final_img),
		f"video": wandb.Video(gif_path)},
		step=len(training.loss_hist))

	run.finish()

def final_plain():
	""" In this experiment we just run the standard training algorithm 
	on all the final test images from the image database. """

	config = Config()
	config.layer1_size = 256
	config.num_channels = 15
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
		build_and_train("final_compare_gradual", config)

def final_center_growing():
	""" In this experiment we compare various center growing squares """

	config = Config()
	config.layer1_size = 256
	config.num_channels = 15
	config.target_channels = 3
	config.target_loss = 0.01
	config.lifetime = 32
	config.size = 32
	config.initial_state = 'sconf_center_black_dot'
	config.edge_strategy = 'EdgeStrategy.TF_SAME'
	config.growing_jump = 10

	for path in glob.glob("images/final/*.png"):
		img_name = os.path.basename(path)
		config.target_state = f'sconf_image("final/{img_name}")'
		build_and_train("final_compare_gradual", config)

def final_stacked_compare():
	""" Compares stacked learning with plain learning
	separately. """

	config = Config()
	config.layer1_size = 256
	config.num_channels = 15
	config.target_channels = 3
	config.target_loss = 0.01
	config.lifetime = 32
	config.size = 32
	config.initial_state = 'sconf_center_black_dot'
	config.edge_strategy = 'EdgeStrategy.TF_SAME'
	config.growing_jump = 10

	imgs = list(glob.glob("images/final/*.png"))

	for path1 in imgs:

		img1 = "final/" + os.path.basename(path1)
		config.target_channels = 3
		config.target_state = f'sconf_image("{img1}")'
		build_and_train('final_compare_stacked', config)

		for path2 in imgs:
			img2 = "final/" + os.path.basename(path2)
			config.target_channels = 6
			config.target_state = f'sconf_imagestack("{img1}", "{img2}")'
			build_and_train('final_compare_stacked', config)

def final_gradual_compare_large():
	""" In this experiment we compare various center growing squares """

	config = Config()
	config.layer1_size = 256
	config.num_channels = 15
	config.target_channels = 3
	config.size = 50
	config.target_loss = 0.025
	config.learning_rate = 3.5e-3
	config.initial_state = 'sconf_center_black_dot'
	config.edge_strategy = 'EdgeStrategy.TF_SAME'

	while True:
		for path in glob.glob("images/final/*.png"):
			img_name = os.path.basename(path)
			for i in [0, 10, 25]:
				config.target_state = f'sconf_image("final/{img_name}")'
				config.growing_jump = i
				build_and_train("compare_gradual_10", config)

def companion_training():
	config = Config()
	config.layer1_size = 256
	config.num_channels = 15
	config.target_channels = 3
	config.target_loss = 0.025
	config.size = 60
	config.initial_state = 'sconf_center_black_dot'
	config.edge_strategy = 'EdgeStrategy.TF_SAME'

	config.target_channels = 3
	config.target_state = f'sconf_image("hard/water.png")'
	build_and_train('companion_2', config)
	config.target_state = f'sconf_image("hard/arch.png")'
	build_and_train('companion_2', config)
	config.target_state = f'sconf_image("hard/pills.png")'
	build_and_train('companion_2', config)

	config.target_channels = 6
	config.target_state = f'sconf_imagestack("hard/water.png", "hard/arch.png")'
	build_and_train('companion_2', config)
	config.target_state = f'sconf_imagestack("hard/water.png", "hard/pills.png")'
	build_and_train('companion_2', config)

	# imgs = list(glob.glob("images/hard/*.png"))

	# for path1 in imgs:

	# 	img1 = "hard/" + os.path.basename(path1)
	# 	config.target_channels = 3
	# 	config.target_state = f'sconf_image("{img1}")'
	# 	build_and_train('companion_1', config)

	# 	for path2 in imgs:
	# 		img2 = "hard/" + os.path.basename(path2)
	# 		config.target_channels = 6
	# 		config.target_state = f'sconf_imagestack("{img1}", "{img2}")'
	# 		build_and_train('companion_1', config)

def kernel_set_compare():
	config = Config()
	config.layer1_size = 256
	config.num_channels = 15
	config.target_channels = 3
	config.target_loss = 0.01
	config.size = 30
	config.initial_state = 'sconf_center_black_dot'
	config.edge_strategy = 'EdgeStrategy.TF_SAME'
	
	for path in glob.glob("images/final/*.png"):
		img_name = os.path.basename(path)
		config.target_state = f'sconf_image("final/{img_name}")'
		for kernel_set in ["kernel_sobel()", "kernel_neighbors()"]:
			config.kernel_set = kernel_set
			build_and_train("compare_kernel_sets", config)

def channel_count_compare():
	config = Config()
	config.layer1_size = 256
	config.target_channels = 3
	config.target_loss = 0.01
	config.size = 20
	config.initial_state = 'sconf_center_black_dot'
	config.edge_strategy = 'EdgeStrategy.TF_SAME'
	
	for path in glob.glob("images/hard/*.png"):
		img_name = os.path.basename(path)
		config.target_state = f'sconf_image("hard/{img_name}")'
		for c in [3, 4, 5, 8, 12, 15, 18, 21, 27]:
			config.num_channels = c
			build_and_train("compare_channel_count_2", config)

def size_compare():
	config = Config()
	config.layer1_size = 256
	config.target_channels = 3
	config.target_loss = 0.01
	config.num_channels = 15
	config.initial_state = 'sconf_center_black_dot'
	config.edge_strategy = 'EdgeStrategy.TF_SAME'
	
	for path in glob.glob("images/hard/*.png"):
		img_name = os.path.basename(path)
		config.target_state = f'sconf_image("hard/{img_name}")'
		for s in [5,10,15,20,25,30,40,50,60]:
			config.size = s
			build_and_train("compare_size", config)

def network_compare():
	config = Config()
	config.two_layers = False
	config.target_channels = 3
	config.target_loss = 0.01
	config.num_channels = 15
	config.initial_state = 'sconf_center_black_dot'
	config.edge_strategy = 'EdgeStrategy.TF_SAME'
	
	for path in glob.glob("images/hard/*.png"):
		img_name = os.path.basename(path)
		config.target_state = f'sconf_image("hard/{img_name}")'
		for s in [5,10,20,30,40,50]:
			config.size = s
			for w in [32,64,128,256,512,1024]:
				config.layer1_size = w
				config.two_layers = False
				build_and_train("network_compare", config)
				config.two_layers = True
				build_and_train("network_compare", config)

def lr_compare():
	config = Config()
	config.two_layers = False
	config.target_channels = 3
	config.target_loss = 0.01
	config.num_channels = 15
	config.initial_state = 'sconf_center_black_dot'
	config.edge_strategy = 'EdgeStrategy.TF_SAME'
	
	for path in glob.glob("images/hard/*.png"):
		img_name = os.path.basename(path)
		config.target_state = f'sconf_image("hard/{img_name}")'
		for l in [1.0e-3, 2.0e-3, 3.5e-3, 4.0e-3, 5.0e-3]:
			config.learning_rate = l
			build_and_train("lr_compare_2", config)

def edge_strategies():
	config = Config()
	config.two_layers = False
	config.target_channels = 3
	config.target_loss = 0.01
	config.num_channels = 15
	config.size = 32
	config.initial_state = 'sconf_center_black_dot'
	
	for path in glob.glob("images/hard/*.png"):
		img_name = os.path.basename(path)
		config.target_state = f'sconf_image("hard/{img_name}")'
		for e in ["EdgeStrategy.TF_SAME", "EdgeStrategy.ZEROS", "EdgeStrategy.ONES", 
		"EdgeStrategy.TORUS", "EdgeStrategy.MIRROR", "EdgeStrategy.RANDOM"]:
			config.edge_strategy = e
			build_and_train("edge_strategies_2", config)

def loss_functions():
	config = Config()
	config.two_layers = False
	config.target_channels = 3
	config.target_loss = 0.01
	config.num_channels = 15
	config.size = 32
	config.initial_state = 'sconf_center_black_dot'
	config.edge_strategy = 'EdgeStrategy.TF_SAME'
	
	for path in glob.glob("images/final/*.png"):
		img_name = os.path.basename(path)
		config.target_state = f'sconf_image("final/{img_name}")'
		config.laplace_loss = False
		build_and_train("final_loss_functions", config)
		config.laplace_loss = True
		build_and_train("final_loss_functions", config)

def initial_states():
	config = Config()
	config.two_layers = False
	config.target_channels = 3
	config.target_loss = 0.01
	config.num_channels = 15
	config.size = 32
	config.initial_state = 'sconf_center_black_dot'
	config.edge_strategy = 'EdgeStrategy.MIRROR'
	
	for path in glob.glob("images/final/*.png"):
		img_name = os.path.basename(path)
		config.target_state = f'sconf_image("final/{img_name}")'
		for path2 in glob.glob("images/initials/*.png"):
			img_name2 = os.path.basename(path2)
			config.initial_state = f'sconf_image("initials/{img_name2}")'
			build_and_train("final_initial_states", config)

def main():
	while True:
		initial_states()

if __name__ == "__main__":
	main()