from os import sep
import module1 as lib

IMAGES = ["lenna.png", "checkers.png", "microbe.png"]
TARGET_LOSS = 0.01

def trial_stacked() -> lib.Training:
	print("Running stacked learning trial...")
	ca = lib.CellularAutomata(
		img_size=16, 
		channel_count=9,
		layer_counts=[128],
		perception_kernel=lib.sobel_state_kernel())
	ca.edge_strategy = lib.EdgeStrategy.ZEROS

	x0 = lambda: ca.pointfilled(ca.constfilled(1.0), 0.0)
	xf = lambda: ca.imagestackfilled(IMAGES)
	xf_val = xf()

	training = lib.init_training(ca, learning_rate=1.5e-3)
	training.run(x0, xf, 24, target_loss=TARGET_LOSS, max_plateau_len=100,
		loss_f=lambda x: lib.loss_all_channels(x, xf_val))
	training.show_sample_run(x0, xf, 24)
	training.show_loss_history()
	return training

def trial_single(index: int) -> lib.Training:
	print("Running single layer learning trial...")
	ca = lib.CellularAutomata(
		img_size=16, 
		channel_count=3,
		layer_counts=[128],
		perception_kernel=lib.sobel_state_kernel())
	ca.edge_strategy = lib.EdgeStrategy.ZEROS

	x0 = lambda: ca.pointfilled(ca.constfilled(1.0), 0.0)
	xf = lambda: ca.imagefilled(IMAGES[index])

	training = lib.init_training(ca, learning_rate=1.5e-3)
	training.run(x0, xf, 24, target_loss=TARGET_LOSS, max_plateau_len=100)
	training.show_sample_run(x0, xf, 24)
	training.show_loss_history()
	return training

def experiment() -> None:
	t2 = trial_single(index=0)
	t3 = trial_single(index=1)
	t4 = trial_single(index=2)
	t1 = trial_stacked()

	parallel = len(t1.loss_hist)
	separate = len(t2.loss_hist) + len(t3.loss_hist) + len(t4.loss_hist)
	print("Parallel: ", parallel)
	print("Separate: ", separate)