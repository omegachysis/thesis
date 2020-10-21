import os
import wandb
import math
import PIL
import io

from training import *

class BioConfig(object):
	def __init__(self):
		self.chain_len = 200
		self.num_channels = 6
		self.hidden_layer_size = 64
		self.learning_rate = 3.5e-3
		self.epsilon = 1.0e-5
		self.training_seconds = 60
		self.num_sample_runs = 3
		self.edge_strategy = 'EdgeStrategy.ZEROS'
		self.initial_state = 'sconf_zero_everywhere'
		self.target_state = 'sconf_random'
		self.loss_fn = 'loss_mse'
		self.lifetime = 200
		self.clamp_values = False
		self.target_loss = 0.0

def derivative_kernel(num_channels: int):
	identify = np.float32([[0,1,0]])
	derivative = np.float32([[-1,0,1]])
	stacked = tf.stack([identify, derivative], axis=-1)
	stacked = stacked[:, :, None]
	kernel = tf.repeat(stacked, repeats=num_channels, axis=2)
	return kernel

class BioCa(tf.keras.Model):
	def __init__(self, config: BioConfig):
		super().__init__()
		self.chain_len = config.chain_len
		self.num_channels = config.num_channels
		self.perception_kernel = derivative_kernel(config.num_channels)

		# Create the input layer.
		inputs = tf.keras.Input(
			shape=(1, self.chain_len, self.num_channels * self.perception_kernel.shape[-1]), 
			dtype=tf.float32)

		conv_layer = tf.keras.layers.Conv2D(
			filters=config.hidden_layer_size, kernel_size=1, activation=tf.nn.relu)
		conv_processed = conv_layer(inputs)

		# Create the output layer.
		output_layer = tf.keras.layers.Conv2D(filters=self.num_channels, kernel_size=1,
			activation=None, kernel_initializer=tf.zeros_initializer())
		outputs = output_layer(conv_processed)

		self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

	def pad_repeat(self, tensor):
		multiples = [3, 3]
		t1 = tf.tile(tensor, multiples)
		w = len(tensor)
		return t1[w-1 : w-1+w+2, w-1 : w-1+w+2]

	@tf.function
	def perceive(self, x):
		conv = tf.nn.depthwise_conv2d(x, 
			filter=self.perception_kernel, 
			strides=[1, 1, 1, 1],
			padding="SAME")
		return conv

	@tf.function
	def call(self, x):
		s = self.perceive(x)
		dx = self.model(s)
		x += dx
		return x

	def randomfilled(self):
		return np.random.rand(1, self.chain_len, self.num_channels).astype(np.float32)

	def constfilled(self, u):
		""" Fills the world with u. """
		return np.ones((1, self.chain_len, self.num_channels), dtype=np.float32) * u

	def to_image(self, x, scale=1):
		vsize = math.ceil(self.num_channels / 3)
		arr = np.zeros((vsize, self.chain_len, 3))
		# Fill the image array with the RGB images generated from the 
		# state in RGB and the hidden channels in groups of 3.
		for i in range(self.num_channels):
			s = x[..., i]
			if i >= 3:
				# Hidden channel, scale to fit color space.
				a = np.min(s)
				b = np.max(s)
				ba = max(1.0, b-a)
				s -= a
				s *= 1/ba
			arr[i//3, :, i%3] = s

		rgb_array = np.uint8(arr * 255.0)
		for dim in range(2):
				rgb_array = np.repeat(rgb_array, scale, dim)
		return PIL.Image.fromarray(rgb_array)
	
	def create_gif(self, xs):
		out = io.BytesIO()
		imgs = [self.to_image(x, scale=3) for x in xs]
		durs = [50 for _ in imgs]
		durs[0] = 500
		durs[-1] = 500
		imgs[0].save(out, 'gif', save_all=True, append_images=imgs[1:], loop=0, duration=durs)
		return out.getvalue()

def loss_mse(target):
	def f(x):
		a = x[..., :3]
		b = target[..., :3]
		return tf.reduce_mean(tf.square(a - b))
	return f

def sconf_zero_everywhere(ca: BioCa):
	return ca.constfilled(0.0)

def sconf_one_everywhere(ca: BioCa):
	return ca.constfilled(1.0)

def sconf_random(ca: BioCa):
	return ca.randomfilled()

def main():
	config = BioConfig()

	wandb.init(project="neural-ca-bioinfo", group="exploring1", config=vars(config))
	ca = BioCa(config)
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