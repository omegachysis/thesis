#%%
import tensorflow as tf
import numpy as np

x = np.float32([[[[0,0], [10,1], [10,4], [10,7], [0,0]]]])
print(x.shape)

filter1 = np.float32([[0,1,0]])
filter2 = np.float32([[-1,0,1]])
filters = tf.stack([filter1, filter2], axis=-1)
filters = filters[:, :, None]
filters = tf.repeat(filters, repeats=2, axis=2)
print(filters.shape)

tf.nn.depthwise_conv2d(input=x, filter=filters, strides=[1,1,1,1], padding="VALID")
# %%
