import time
from random import random

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython import display
from tensorflow.examples.tutorials.mnist import input_data
class SimpleConvAutoEncoder(tf.keras.Model):
    def __init__(self):
        super(SimpleConvAutoEncoder, self).__init__()
        self.encoder = tf.keras.Sequential(
            [
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation=tf.nn.relu),
            tf.keras.layers.Dense(10) # no activation
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(10,)),
                tf.keras.layers.Dense(units=10),
                tf.keras.layers.Dense(units=512, activation=tf.nn.relu),
                tf.keras.layers.Dense(units=1568, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(7,7,32)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64,
                    kernel_size=3,
                    strides=(2,2),
                    padding = 'SAME',
                    activation='relu'
                ),
                tf.keras.layers.Conv2DTranspose(
                    filters=32,
                    kernel_size=3,
                    strides=(2, 2),
                    padding='SAME',
                    activation='relu'
                ),
                tf.keras.layers.Conv2DTranspose(
                    filters=1,
                    kernel_size=3,
                    strides=(1, 1),
                    padding='SAME',
                ),
            ]
        )
    def Encode(self, x):
        return self.encoder(x)

    def Decode(self, x):
        logits = self.decoder(x)
        return tf.sigmoid(logits)


def compute_apply_gradients(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def compute_loss(model : SimpleConvAutoEncoder,x_real):
    z = model.Encode(x_real)
    logits = model.Decode(z)
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=x_real)

def train_and_save_results(train_set, test_set):
    epochs = 10
    optimizer =tf.optimizers.Adam()(1e-4)
    model = SimpleConvAutoEncoder()
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        for train_x in train_set:
            compute_apply_gradients(model, train_x, optimizer)
        end_time = time.time()
        loss = np.inf
        if epoch:
            for test_x in test_set:
                loss = compute_loss(model, test_x)
            display.clear_output(wait=False)
            print('Epoch: {}, Test set ELBO: {}, '
                  'time elapse for current epoch {}'.format(epoch,
                                                            loss,
                                                            end_time - start_time))
        generate_and_save_images(model, epoch, test_set[random.randint(0,len(test_set))])
def generate_and_save_images(model, epoch, test_input):
  predictions = model.Decode(test_input)
  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0], cmap='gray')
      plt.axis('off')

  # tight_layout minimizes the overlap between 2 sub-plots
  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()

def load_data():
    TRAIN_BUF = 60000
    BATCH_SIZE = 100
    TEST_BUF = 10000
    (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')

    # Normalizing the images to the range of [0., 1.]
    train_images /= 255.
    test_images /= 255.
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_images).shuffle(TEST_BUF).batch(BATCH_SIZE)
    return train_dataset ,test_dataset

if __name__ == '__main__':
    tf.enable_eager_execution()
    train_dataset, test_dataset = load_data()
    train_and_save_results(train_dataset, test_dataset)
