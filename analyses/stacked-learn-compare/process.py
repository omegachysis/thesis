from math import sqrt
import csv
from os import stat_result
from scipy import stats
import statistics

non_stacked_times = {}
stacked_times = {}

with open("data.csv") as csvfile:
	reader = csv.reader(csvfile)
	header = next(reader)
		# "Name","total_seconds","target_state"
	target_col = header.index("target_state")
	time_col = header.index("total_seconds")
	for row in reader:
		target = row[target_col]
		time = row[time_col]
		if time == '': continue
		time = float(time)
		
		if target.startswith("sconf_imagestack("):
			img1, img2 = eval(target.replace("sconf_imagestack", ""))
			if (img1, img2) not in stacked_times:
				stacked_times[(img1, img2)] = []
			stacked_times[(img1, img2)].append(time)
		elif target.startswith("sconf_image("):
			img = eval(target.replace("sconf_image",""))
			if img not in non_stacked_times:
				non_stacked_times[img] = []
			non_stacked_times[img].append(time)
		else:
			raise Exception()

non_stacked_avgs = {}
for k,v in non_stacked_times.items():
	non_stacked_avgs[k] = statistics.mean(v)

stacked_avgs = {}
for k,v in stacked_times.items():
	stacked_avgs[k] = statistics.mean(v)

# Check symmetry of stacked learning.
asym_dist1 = []
asym_dist2 = []
for k,v in stacked_avgs.items():
	img1, img2 = k
	if img1 > img2: continue
	if (img2, img1) in stacked_avgs:
		asym_dist1.append(v)
		asym_dist2.append(stacked_avgs[(img2, img1)])

asym_diffs = [i - j for (i,j) in zip(asym_dist1, asym_dist2)]
print("Asymmetric differences:")
print(stats.describe(asym_diffs))
print("Correlation of asymmetries:")
print(stats.pearsonr(asym_dist1, asym_dist2))

# Analyze speed difference between stacked and single combined.
non_stacked_dist = []
stacked_dist = []
for k,v in stacked_avgs.items():
	img1, img2 = k
	sep_time = non_stacked_avgs[img1] + non_stacked_avgs[img2]
	non_stacked_dist.append(sep_time)
	stacked_dist.append(v)

print("-" * 80)
print("Stats for non stacked learning:")
print(stats.describe(non_stacked_dist))
print("Stats for stacked learning:")
print(stats.describe(stacked_dist))
print("T test:")
print(stats.ttest_ind(non_stacked_dist, stacked_dist, equal_var=False))

print("-" * 80)
print("Averages across images in stacks")
dist1 = []
dist2 = []

for img, sep_time in non_stacked_avgs.items():
	times = []
	for k,v in stacked_avgs.items():
		if img in k:
			times.append(v)
	
	dist1.append(sep_time)
	dist2.append(statistics.mean(times))
	print(img, sep_time, stats.describe(times))

print("-" * 80)
print("Correlation between separate time and average time in stacks:")
print(stats.pearsonr(dist1, dist2))