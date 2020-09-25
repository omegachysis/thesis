import module1 as lib
import matplotlib
import matplotlib.pyplot as plt
from typing import Dict

# Try different placements of nucleation sites around an image.
# This is to see if the local complexity of the image might have something 
# to do with how well the nucleation site works.
IMG_SIZE = 24
LIFETIME = 40
TRIAL_SECONDS = 60
NUM_TRIALS = 1
NUM_TEST_POINTS = 4

class TrialDescription:
	trial_num: int
	nucleation_x: float
	nucleation_y: float
	
	def __str__(self) -> str:
		return f"{self.trial_num},{self.nucleation_x},{self.nucleation_y}"

def trial(trial: TrialDescription) -> lib.Training:
	print("\nRunning trial: ", trial)
	ca = lib.CellularAutomata(
		img_size=IMG_SIZE, channel_count=6, layer_counts=[64],
		perception_kernel=lib.sobel_state_kernel())
	ca.edge_strategy = lib.EdgeStrategy.TORUS
	
	x0 = lambda: ca.pointfilled(
		ca.constfilled(1.0), point_value=0.0, pos=(trial.nucleation_x, trial.nucleation_y))
	xf = lambda: ca.imagefilled("lenna_circle.png")

	training = lib.init_training(ca, learning_rate=1.0e-3)
	training.run(x0, xf, lifetime=LIFETIME, max_seconds=TRIAL_SECONDS, lock_release=LIFETIME//2)
	xs = training.show_sample_run(x0, xf, lifetime=LIFETIME, lock_release=LIFETIME//2)
	training.show_loss_history()
	training.save(name=str(trial), sample_run_xs=xs)
	return training

Results = Dict[TrialDescription, lib.Training]

def save_results(results: Results) -> None:
	with open("nucleation_results.csv", 'w') as f:
		f.write("Trial number, Nucleation X, Nucleation Y, Min Loss \n")
		for trial, training in results.items():
			csv = str(trial) + ','
			csv += str(min(training.loss_hist))
			f.write(csv + '\n')

def experiment() -> Results:
	res: Results = {}
	for x in range(NUM_TEST_POINTS):
		for y in range(NUM_TEST_POINTS):
			for i in range(NUM_TRIALS):
				des = TrialDescription()
				des.nucleation_x = x / NUM_TEST_POINTS
				des.nucleation_y = y / NUM_TEST_POINTS
				des.trial_num = i
				res[des] = trial(des)
				save_results(res)
	return res

def analysis():
	data = []
	with open("nucleation_results.csv", 'r') as f:
		row = []
		trial_set = []
		for line in f.readlines()[1:]:
			min_loss = float(line.split(',')[3])
			trial_set.append(min_loss)
			if len(trial_set) == NUM_TRIALS:
				avg_loss = sum(trial_set) / NUM_TRIALS
				row.append(avg_loss)
				if len(row) == NUM_TEST_POINTS:
					data.append(row)
					row = []
				trial_set = []
	
	plt.imshow(data)
	plt.colorbar()
	plt.savefig("nucleation_placement_plot.png")
	plt.show()