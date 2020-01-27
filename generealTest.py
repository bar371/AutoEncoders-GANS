import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', validation_size=0)

img = mnist.train.images[2]
plt.imshow(img.reshape((28, 28)), cmap='Greys_r')


learning_rate = 0.001
# Input and target placeholders
inputs_ = tf.placeholder(tf.float32, (None, 28,28,1), name="input")
targets_ = tf.placeholder(tf.float32, (None, 28,28,1), name="target")

### Encoder
conv1 = tf.layers.conv2d(inputs=inputs_, strides = (2,2) , filters=32, kernel_size=3, padding='same', activation=tf.nn.relu)
# Now 14x14x32
conv2 = tf.layers.conv2d(inputs=conv1, strides = (2,2) , filters=64, kernel_size=3, padding='same', activation=tf.nn.relu)
# Now 7x7x64
flattend = tf.layers.flatten(conv2)
# Now 3136x1
dense1 = tf.layers.dense(flattend, units=512, activation=tf.nn.relu)
# dense2 = tf.layers.dense(dense1, units=10, activation=tf.nn.relu)
encoded = dense1
d2 = PCA(encoded)

### Decoder
# dense2up = tf.layers.dense(encoded, units=(50,), activation=tf.nn.relu)
dense1up = tf.layers.dense(encoded, units=7*7*32, activation=tf.nn.relu)
reshaped = tf.reshape(dense1up, shape=(10,7,7,32))
# Now 7x7x64
conv4 = tf.layers.conv2d_transpose(inputs=reshaped, strides=(2,2), filters=64, kernel_size=3, padding='same', activation=tf.nn.relu)
# Now 7x7x64
# Now 14x14x8
conv5 = tf.layers.conv2d_transpose(inputs=conv4,strides = (2,2), filters=32, kernel_size=3, padding='same', activation=tf.nn.relu)
# Now 28x28x32
logits = tf.layers.conv2d_transpose(inputs=conv5, filters=1, kernel_size=3, padding='same')
# Now 28x28x1

#Now 28x28x1

# Pass logits through sigmoid to get reconstructed image
decoded = tf.nn.sigmoid(logits)

# Pass logits through sigmoid and calculate the cross-entropy loss
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_, logits=logits)

# Get cost and define the optimizer
cost = tf.reduce_mean(loss)
opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# sess = tf.Session()
# epochs = 10
# batch_size = 10
# sess.run(tf.global_variables_initializer())
# for e in range(epochs):
#     for ii in range(mnist.train.num_examples//batch_size):
#         batch = mnist.train.next_batch(batch_size)
#         imgs = batch[0].reshape((-1, 28, 28, 1))
#         batch_cost, _ = sess.run([cost, opt], feed_dict={inputs_: imgs,
#                                                          targets_: imgs})
#
#         print("Epoch: {}/{}...".format(e+1, epochs),
#               "Training loss: {:.4f}".format(batch_cost))
# print('training done ')
#
#
# fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(20,4))
# in_imgs = mnist.test.images[:10]
# reconstructed = sess.run(decoded, feed_dict={inputs_: in_imgs.reshape((10, 28, 28, 1))})
# print('done recon')
# for images, row in zip([in_imgs, reconstructed], axes):
#     for img, ax in zip(images, row):
#         ax.imshow(img.reshape((28, 28)), cmap='Greys_r')
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
# plt.show()
sess = tf.Session()
epochs = 3
batch_size = 10
# Set's how much noise we're adding to the MNIST images
noise_factor = 0.5
sess.run(tf.global_variables_initializer())
for e in range(epochs):
    for ii in range(mnist.train.num_examples // batch_size):
        batch = mnist.train.next_batch(batch_size)
        # Get images from the batch
        imgs = batch[0].reshape((-1, 28, 28, 1))

        # Add random noise to the input images
        noisy_imgs = imgs + noise_factor * np.random.randn(*imgs.shape)
        # Clip the images to be between 0 and 1
        noisy_imgs = np.clip(noisy_imgs, 0., 1.)

        # Noisy images as inputs, original images as targets
        batch_cost, _ = sess.run([cost, opt], feed_dict={inputs_: noisy_imgs,
                                                         targets_: imgs})

        print("Epoch: {}/{}...".format(e + 1, epochs),
              "Training loss: {:.4f}".format(batch_cost))

fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(20,4))
in_imgs = mnist.test.images[:10]
noisy_imgs = in_imgs + noise_factor * np.random.randn(*in_imgs.shape)
noisy_imgs = np.clip(noisy_imgs, 0., 1.)

reconstructed = sess.run(decoded, feed_dict={inputs_: noisy_imgs.reshape((10, 28, 28, 1))})

for images, row in zip([noisy_imgs, reconstructed], axes):
    for img, ax in zip(images, row):
        ax.imshow(img.reshape((28, 28)), cmap='Greys_r')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

fig.tight_layout(pad=0.1)
plt.show()