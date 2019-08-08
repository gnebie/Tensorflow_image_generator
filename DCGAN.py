from __future__ import absolute_import, division, print_function, unicode_literals

# !pip install -q tensorflow-gpu==2.0.0-beta1

import tensorflow as tf

tf.__version__

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

from IPython import display


def get_train_images():
	# get the images from tensorflow exemples and
	(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

	train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
	train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]
	return train_images

class CheckpointClass():
	def __init__(self, generator_optimizer, discriminator_optimizer, generator, discriminator):
		self.checkpoint_dir = './training_checkpoints'
		self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
		self.checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
		                                 discriminator_optimizer=discriminator_optimizer,
		                                 generator=generator,
		                                 discriminator=discriminator)

	def save():
		self.checkpoint.save(file_prefix = self.checkpoint_prefix)

	def restore_latest():
		self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))


class DCGAN():
	def __init__(self):
		print("test")
		# get the train images
		train_images = get_train_images()

		epoch_dir = "epoch"

		BUFFER_SIZE = 60000
		self.BATCH_SIZE = 256

		# create a directory for the images
		os.makedirs(epoch_dir, exist_ok=True)

		# Batch and shuffle the data
		train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(self.BATCH_SIZE)

		self.generator = make_generator_model()

		noise = tf.random.normal([1, 100])
		generated_image = self.generator(noise, training=False)

		plt.imshow(generated_image[0, :, :, 0], cmap='gray')

		self.discriminator = make_discriminator_model()
		decision = self.discriminator(generated_image)
		print (decision)
		self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

		self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
		self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

		self.check = CheckpointClass(self.generator_optimizer, self.discriminator_optimizer, self.generator, self.discriminator)

		self.EPOCHS = 50
		self.noise_dim = 100
		num_examples_to_generate = 16

		# We will reuse this seed overtime (so it's easier)
		# to visualize progress in the animated GIF)
		self.seed = tf.random.normal([num_examples_to_generate, self.noise_dim])

		self.train(train_dataset, self.EPOCHS)

		self.check.restore_latest()

		display_image(self.EPOCHS)


	# Notice the use of `tf.function`
	# This annotation causes the function to be "compiled".
	@tf.function
	def train_step(self, images):
	    noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])

	    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
	      generated_images = self.generator(noise, training=True)

	      real_output = self.discriminator(images, training=True)
	      fake_output = self.discriminator(generated_images, training=True)

	      gen_loss = generator_loss(fake_output, self.cross_entropy)
	      disc_loss = discriminator_loss(real_output, fake_output, self.cross_entropy)

	    gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
	    gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

	    self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
	    self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

	def train(self, dataset, epochs):
	  for epoch in range(epochs):
	    start = time.time()

	    for image_batch in dataset:
	      self.train_step(image_batch)

	    # Produce images for the GIF as we go
	    display.clear_output(wait=True)
	    generate_and_save_images(self.generator,
	                             epoch + 1,
	                             self.seed)

	    # Save the model every 15 epochs
	    if (epoch + 1) % 15 == 0:
	      self.check.save()

	    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

	  # Generate after the final epoch
	  display.clear_output(wait=True)
	  generate_and_save_images(self.generator,
	                           epochs,
	                           self.seed)



# first neural network
def make_generator_model():
	# TODO need to see more about the layers
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model

# second neural network
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


# This method returns a helper function to compute cross entropy loss

def discriminator_loss(real_output, fake_output, cross_entropy):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output, cross_entropy):
    return cross_entropy(tf.ones_like(fake_output), fake_output)



def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig(os.join(epoch_dir, 'image_at_epoch_{:04d}.png'.format(epoch)))
  # plt.show()


# Display a single image using the epoch number
def display_image(epoch_no):
  return PIL.Image.open(os.join(epoch_dir, 'image_at_epoch_{:04d}.png'.format(epoch_no)))


def create_gif():
	anim_file = 'dcgan.gif'

	with imageio.get_writer(anim_file, mode='I') as writer:
	  filenames = glob.glob('image*.png')
	  filenames = sorted(filenames)
	  last = -1
	  for i,filename in enumerate(filenames):
	    frame = 2*(i**0.5)
	    if round(frame) > round(last):
	      last = frame
	    else:
	      continue
	    image = imageio.imread(filename)
	    writer.append_data(image)
	  image = imageio.imread(filename)
	  writer.append_data(image)

	import IPython
	if IPython.version_info > (6,2,0,''):
	  display.Image(filename=anim_file)



def main():
	dcgan = DCGAN()


if __name__ == '__main__':
    main()
