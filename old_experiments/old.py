def loss_mean_square(x, target, mask=None):
	if mask is not None:
		return tf.reduce_mean(mask[...,:3] * tf.square(x[...,:3] - target[...,:3]))
	else:
		return tf.reduce_mean(tf.square(x[...,:3] - target[...,:3]))

def loss_all_channels(x, target):
	return tf.reduce_mean(tf.square(x - target))

def loss_harmonize(x):
	channel_count = x.shape[3]
	Δ = tf.reshape(tf.constant([
			[1/4, 1/2, 1/4],
			[1/2, -3,  1/2],
			[1/4, 1/2, 1/4]
	]), shape=[3,3,1])
	Δ = Δ[:,:,None,:]
	Δ = tf.repeat(Δ, repeats=channel_count, axis=2)
	Δx = tf.nn.depthwise_conv2d(x, Δ, strides=[1,1,1,1], padding="VALID")[0]
	return tf.reduce_mean(tf.square(Δx))