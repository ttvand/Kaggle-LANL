# Basic VAE MNIST - utilities
# Adapted from https://keras.io/examples/variational_autoencoder_deconv/
import keras
from keras import backend as K
from keras.layers import Conv1D
from keras.layers import Conv2DTranspose
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import Reshape
from keras.losses import mse
from keras.models import Model

import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.decomposition import PCA


# LANL data generator
def lanl_vae_generator(features, batch_size):
  steps_per_epoch=features.shape[0] // batch_size
  while True:
    shuffled_ids = np.random.permutation(features.shape[0])
    for i in range(steps_per_epoch):
      batch_features = features[shuffled_ids[i*batch_size:((i+1)*batch_size)]]
      
      # One input, no output
      yield [batch_features], [] 


# Reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + z_std*eps
def sample_gaussian(args):
  z_mean, z_std = args
  batch = K.shape(z_mean)[0]
  dim = K.int_shape(z_mean)[1]
  epsilon = K.random_normal(shape=(batch, dim))
  return z_mean + z_std * epsilon


# VAE loss - reconstruction + beta*kl_loss
def vae_loss(z_mean, z_std, inputs, outputs, num_subchunks, num_features,
             hyperpars):
  rec_loss = mse(K.flatten(inputs), K.flatten(outputs))
  rec_loss = K.mean(rec_loss*num_subchunks*num_features)
  rec_loss = Lambda(lambda x: x, name='reconstruction_loss')(rec_loss)
  
  kl_loss = 1 + K.log(K.square(z_std)) - K.square(z_mean) - K.square(z_std)
#  kl_loss = 1/2*(K.square(z_std) + K.square(z_mean) - 2*K.log(z_std) - 1)
  kl_loss = K.mean(K.sum(kl_loss, axis=-1)*-1/hyperpars['latent_dim'])
  kl_loss *= hyperpars['kl_beta']
  kl_loss = Lambda(lambda x: x, name='kl_loss')(kl_loss)
  
  metrics = {'reconstruction_loss': rec_loss, 'kl_loss': kl_loss}
  
  return [rec_loss, kl_loss], metrics


# Custom 1D conv transpose
# Source: https://stackoverflow.com/questions/44061208/how-to-implement-the-conv1dtranspose-in-keras
def Conv1DTranspose(input_tensor, filters, kernel_size, strides=1,
                    padding='same', activation=None, name=None):
  x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
  x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1),
                      strides=(strides, 1), padding=padding,
                      activation=activation)(x)
  x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
  if name is not None:
    x = Lambda(lambda x: x, name=name)(x)
  return x


# MNIST autoencoder model
def lanl_vae(hyperpars):
  num_subchunks = hyperpars['num_subchunks']
  num_features = hyperpars['num_features']
  # 1) Encoder
  inputs = Input((num_subchunks, num_features), name='image_inputs')
  x = inputs
  for (filters, kernel, strides) in hyperpars['filters_kernels_strides']:
    x = Conv1D(filters=filters, kernel_size=kernel, strides=strides,
               padding='same', activation='relu')(x)
  shape = K.int_shape(x) # shape info needed to build decoder model
  print('Shape after convolution: {}'.format(shape))
  x = Flatten()(x)
  for layer_size in hyperpars['latent_mlp_layers']:
    x = Dense(layer_size, activation='relu')(x)
  z_mean = Dense(hyperpars['latent_dim'], activation='linear')(x)
  z_std = Dense(hyperpars['latent_dim'], activation='softplus')(x)
  latents = Lambda(
      sample_gaussian, output_shape=(hyperpars['latent_dim'],), name='z')(
          [z_mean, z_std])
  encoder = Model(inputs=inputs,
                  outputs=[z_mean, z_std, latents], name='encoder')
  
  # 2) Decoder
  latent_inputs = Input(shape=(hyperpars['latent_dim'],), name='latent_inputs')
  x = latent_inputs
  for layer_size in hyperpars['latent_mlp_layers'][::-1]:
    x = Dense(layer_size, activation='relu')(x)
  x = Dense(shape[1] * shape[2], activation='relu')(x)
  x = Reshape((shape[1], shape[2]))(x)
  for (filters, kernel, strides) in hyperpars['filters_kernels_strides'][::-1]:
    x = Conv1DTranspose(input_tensor=x, filters=filters, kernel_size=kernel,
                        strides=strides, padding='same', activation='relu')
  decoder_outputs = Conv1DTranspose(
      input_tensor=x, filters=num_features, kernel_size=3, strides=1,
      activation=None, padding='same', name='decoder_outputs')
  decoder = Model(inputs=latent_inputs,
                  outputs=decoder_outputs, name='decoder')

  # 3) Combine the encoder and decoder into a Model with a custom loss.
  # Compute loss here because the VAE loss does not follow the standard format
  # See https://stackoverflow.com/questions/50063613/add-loss-function-in-keras
  # first decoder input = third encoder output
  outputs = decoder(encoder(inputs)[2])
  losses, metrics = vae_loss(z_mean, z_std, inputs, outputs, num_subchunks,
                             num_features, hyperpars)  
  
  model = Model(inputs=inputs, outputs=outputs, name='vae')
  model.add_loss(losses)
  
  return (model, encoder, decoder, metrics)


# Hack to add custom metrics when using add_loss
# Credit to May4m from https://github.com/keras-team/keras/issues/9459
def add_custom_metrics(model, custom_metrics):
  for k in custom_metrics:
    model.metrics_names.append(k)
    model.metrics_tensors.append(custom_metrics[k])


# Generate plots of the latent space, reconstructions and random samples.
def plot_results(models, data, hyperpars, batch_size=128,
                 model_folder='vae_mnist_figures'):
  """Plots labels and LANL chunks as a function of 2-dim latent vector."""
  encoder, decoder = models
  train_features, test_features, train_eq_ids, train_ttf = data
  latent_dim = hyperpars['latent_dim']
  num_subchunks = hyperpars['num_subchunks']
  num_features = hyperpars['num_features']

  # 1) Display a 2D plot of the chunk classes in the latent space
  filename = os.path.join(model_folder, "vae_mean.png")
  all_features = np.concatenate([train_features, test_features], 0)
  num_train = train_features.shape[0]
  num_test = test_features.shape[0]
  is_train = np.concatenate([
      0.5+train_eq_ids*np.ones(num_train, dtype=np.bool)/(16*2),
      np.zeros(num_test, dtype=np.bool)])
  z_mean, _, _ = encoder.predict(all_features)
  
  # Apply PCA to the embedded z_mean if the dimensionality > 2
  if latent_dim > 2:
    z_mean = PCA(n_components=2).fit_transform(z_mean)
  
  plt.figure(figsize=(12, 10))
  plt.scatter(z_mean[:, 0], z_mean[:, 1], c=is_train)
  plt.colorbar()
  plt.clim(0, 1)
  plt.xlabel("z[0]")
  plt.ylabel("z[1]")
  plt.xlim(-5, 5)
  plt.ylim(-5, 5)
  plt.savefig(filename)
  plt.show()
  
  # 2) Plot the encodings, colored by time to failure in latent space
  filename = os.path.join(model_folder, "vae_mean_ttf.png")
  plt.figure(figsize=(12, 10))
  plt.scatter(z_mean[:num_train, 0], z_mean[:num_train, 1], c=train_ttf)
  plt.colorbar()
  plt.xlabel("z[0]")
  plt.ylabel("z[1]")
  plt.xlim(-5, 5)
  plt.ylim(-5, 5)
  plt.savefig(filename)
  plt.show()
  
  # 3) Plot the encodings, colored by eq id
  filename = os.path.join(model_folder, "vae_mean_eq_id.png")
  plt.figure(figsize=(12, 10))
  plt.scatter(z_mean[:num_train, 0], z_mean[:num_train, 1],
              c=is_train[:num_train])
  plt.colorbar()
  plt.xlabel("z[0]")
  plt.ylabel("z[1]")
  plt.xlim(-5, 5)
  plt.ylim(-5, 5)
  plt.savefig(filename)
  plt.show()
  
  # 4) Display a 2D plot of the train earthquake cycles in the latent space
  for eq_id in np.unique(train_eq_ids):
    filename = os.path.join(model_folder, "vae_mean_eq_" + str(eq_id) + ".png")
    train_match_eq_ids = np.where(train_eq_ids == eq_id)[0]
    test_samp_ids = num_train + np.random.randint(
        0, num_test, train_match_eq_ids.shape[0])
    is_train_eq = is_train[np.hstack([train_match_eq_ids, test_samp_ids])]
    z_mean_eq = z_mean[np.hstack([train_match_eq_ids, test_samp_ids])]
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean_eq[:, 0], z_mean_eq[:, 1], c=is_train_eq)
    plt.colorbar()
    plt.clim(0, 1)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.savefig(filename)
    plt.show()

#  # 4) Display a 10x3 2D manifold over chunks
#  filename_base = os.path.join(model_folder, "chunks_over_latent")
#  nx = 10
#  ny = 3
#  # linearly spaced coordinates corresponding to the 2D plot
#  # of chunk classes in the latent space
#  grid_x = np.linspace(-4, 4, nx)
#  grid_y = np.linspace(-4, 4, ny)[::-1]
#
#  # Loop over the latent dimensions: freeze all but 2 and sweep over these 2.
#  num_latent_figures = latent_dim // 2
#  for fig_id in range(num_latent_figures):
#    figure = np.zeros((num_subchunks * ny, num_features * nx))
#    z_sample = np.random.normal(size=(1, latent_dim))
#    for i, yi in enumerate(grid_y):
#      for j, xi in enumerate(grid_x):
#        z_sample[0, int(fig_id*2)] = xi
#        z_sample[0, int(fig_id*2 + 1)] = yi
#        x_decoded = decoder.predict([z_sample])
#        chunk = x_decoded[0]
#        figure[i * num_subchunks: (i + 1) * num_subchunks,
#               j * num_features: (j + 1) * num_features] = chunk
#  
#    plt.figure(figsize=(10, 10))
#    plt.xlabel("z[0]")
#    plt.ylabel("z[1]")
#    plt.imshow(figure, cmap='Greys_r')
#    plt.savefig(filename_base + str(fig_id) + '.png')
#    plt.show()
    
  # 5) Display original chunks and reconstructions
  filename = os.path.join(model_folder, "vae_reconstructions.png")
  num_reconstructions = 20
  sample_ids = np.random.choice(range(test_features.shape[0]),
                                num_reconstructions, replace=False)
  
  orig_chunks = test_features[sample_ids]
  _, _, encodings = encoder.predict(orig_chunks)
  reconstructions = decoder.predict(encodings)
  figure = np.zeros((2*num_subchunks, num_features*num_reconstructions))
  figure[:num_subchunks] = np.transpose(orig_chunks, [1, 0, 2]).reshape(
      num_subchunks, -1)
  figure[num_subchunks:] = np.transpose(reconstructions, [1, 0, 2]).reshape(
      num_subchunks, -1)
  plt.figure(figsize=(10, 10))
  plt.imshow(figure, cmap='Greys_r')
  plt.savefig(filename)
  plt.show()
  
  
# Custom Callback for checkpointing a specific model
# Inspired by https://stackoverflow.com/questions/50983008/how-to-save-best-weights-of-the-encoder-part-only-during-auto-encoder-training
# Callback source: https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L633
# Terrible BUG: the main model is saved when calling the second variable model.
class CustomCheckpointer(keras.callbacks.Callback):
  def __init__(self, filepath, custom_model, monitor, mode, save_best_only,
               verbose=0, verbose_description='encoder'):
    self.filepath = filepath
    self.custom_model = custom_model
    self.monitor = monitor
    self.save_best_only = save_best_only
    self.verbose = verbose
    self.description = verbose_description
    
    print('Initializing custom checkpointer for model `{}`.'.format(
        self.custom_model.name))
    self.monitor_op = np.less if mode == 'min' else np.greater
    self.best = np.Inf if mode == 'min' else -np.Inf
  
  def on_epoch_end(self, epoch, logs=None):
    current = logs.get(self.monitor)
    if not self.save_best_only or self.monitor_op(current, self.best):
      if self.verbose > 0:
        print('Saving the custom {} model to {}'.format(
            self.description, self.filepath))
      self.best = current
      self.custom_model.save(self.filepath, overwrite=True)
      
      
# Custom Keras callback for plotting learning progress
class PlotLosses(keras.callbacks.Callback):
  def on_train_begin(self, logs={}):
    self.i = 0
    self.x = []
    self.val_losses = []
    self.fig = plt.figure()
    self.logs = []
    
    loss_extensions = ['', 'reconstruction', 'kl']
    self.best_loss_key = 'loss'
    self.loss_keys = [e + ('_' if e else '') + 'loss' for e in loss_extensions]
    self.losses = {k: [] for k in self.loss_keys}

  def on_epoch_end(self, epoch, logs={}):
    self.logs.append(logs)
    self.x.append(self.i)
    for k in self.loss_keys:
      self.losses[k].append(logs.get(k))
    self.i += 1
    
    best_loss = np.repeat(np.array(self.losses[self.best_loss_key]).min(),
                              self.i).tolist()
    best_id = (1+np.repeat(
        np.array(self.losses[self.best_loss_key]).argmin(), 2)).tolist()
    for k in self.loss_keys:
      plt.plot([1+x for x in self.x], self.losses[k], label=k)
    all_losses = np.array(list(self.losses.values())).flatten()
    if len(self.losses) > 1:
      plt.plot([1+x for x in self.x], best_loss, linestyle="--", color="r",
               label="")
      plt.plot(best_id, [0, best_loss[0]],
               linestyle="--", color="r", label="")
    plt.ylim(0, max(all_losses) + 0.1)
    plt.legend()
    plt.show()