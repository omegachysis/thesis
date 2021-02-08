from math import sqrt
import csv
from scipy import stats

non_grad_times = []
grad_times = []

with open("data.csv") as csvfile:
	reader = csv.reader(csvfile)
	header = next(reader)
		# "Name","Runtime","Notes","Tags","growing_jump","target_state","loss","loss0","step","total_seconds"
	jump_col = header.index("growing_jump")
	target_col = header.index("target_state")
	time_col = header.index("total_seconds")
	for row in reader:
		jump = int(row[jump_col])
		target = row[target_col]
		time = row[time_col]
		if time == '': continue
		time = float(time)
		
		if (jump == 0):
			non_grad_times.append(time)
		else:
			grad_times.append(time)

print("Stats for non gradual learning:")
print(stats.describe(non_grad_times))
print("Stats for gradual learning:")
print(stats.describe(grad_times))

print("T test:")
print(stats.ttest_ind(non_grad_times, grad_times, equal_var=False))