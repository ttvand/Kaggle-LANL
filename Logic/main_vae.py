# VAE of LANL train and test spectrograms
# Adapted from https://keras.io/examples/variational_autoencoder_deconv/
# Inspired by https://wiseodd.github.io/techblog/2016/12/17/conditional-vae/
import math
import os
import pandas as pd
import utils_vae

from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.models import load_model
from keras.utils import plot_model

data_folder = '/media/tom/cbd_drive/Kaggle/LANL/Data/'

# Execution and model (hyper)parameters
mode = ['train', 'inspect', 'train_inspect'][2]
include_cpc_encodings = False
include_freq_features = True
only_fft_freq_features = True
num_subchunks = 100
save_folder = data_folder + 'vae_lanl/'
os.makedirs(save_folder, exist_ok=True)
save_path_encoder = save_folder + 'vae_encoder.h5'
save_path_decoder = save_folder + 'vae_decoder.h5'
hyperpars = {
    'batch_size': 32,
    'num_epochs': 100,
    'initial_lr': 1e-3,
    'reduce_lr_patience': 3,
    'es_patience': 5,
    
    'filters_kernels_strides': [],
#    'filters_kernels_strides': [(32, 3, 2), (16, 3, 2)],
#    'filters_kernels_strides': [(32, 3, 1), (32, 3, 1), (32, 3, 1), (32, 3, 2)],
    'latent_mlp_layers': [64, 16],
    'latent_dim': 2,
    
    'kl_beta': 5, # Beta-VAE https://openreview.net/pdf?id=Sy2fzU9gl
    }
def get_features(features_df, data_cols, num_subchunks):
  features_vals = features_df[data_cols].values
  num_features = features_vals.shape[-1]
  return features_vals.reshape(-1, num_subchunks, num_features)

if 'train' in mode:
  train_encodings_features = pd.read_csv(
      data_folder + 'train_main_cpc_encodings_and_features.csv')
  data_cols = [c for c in train_encodings_features.columns if (
      c != 'target' and not 'notrain' in c)]
  if not include_freq_features:
    data_cols = [c for c in data_cols if not c[:4] =='sub_' in c]
  else:
    if only_fft_freq_features:
      data_cols = [c for c in data_cols if c[:8] =='sub_fft_' in c]
  if not include_cpc_encodings:
    data_cols = [c for c in data_cols if not c[:4] =='enc_' in c]
  train_features = get_features(train_encodings_features, data_cols,
                                num_subchunks)
  train_eq_ids = train_encodings_features.notrain_eq_id.values[::num_subchunks]
  train_ttf = train_encodings_features.target.values[::num_subchunks]
  test_encodings_features = pd.read_csv(
      data_folder + 'test_main_cpc_encodings_and_features.csv')
  test_features = get_features(test_encodings_features, data_cols,
                               num_subchunks)
  
hyperpars['num_features'] = len(data_cols)
hyperpars['num_subchunks'] = num_subchunks
  
if 'train' in mode:
  K.clear_session()
  (model, encoder, decoder, custom_metrics) = utils_vae.lanl_vae(hyperpars)
  adam = Adam(lr=hyperpars['initial_lr'])
  model.compile(optimizer=adam)
  utils_vae.add_custom_metrics(model, custom_metrics)
  plot_model(encoder, to_file=save_folder+'vae_cnn_encoder.png',
             show_shapes=True)
  plot_model(decoder, to_file=save_folder+'vae_cnn_decoder.png',
             show_shapes=True)
  plot_model(model, to_file=save_folder+'vae_cnn.png', show_shapes=True)
  (monitor, monitor_mode) = ('loss', 'min')
  train_gen = utils_vae.lanl_vae_generator(
      train_features, hyperpars['batch_size'])
  earlystopper = EarlyStopping(
      monitor=monitor, mode=monitor_mode,
      patience=hyperpars['es_patience'], verbose=1)
  encoder_checkpointer = utils_vae.CustomCheckpointer(
        save_path_encoder, encoder, monitor=monitor, mode=monitor_mode,
        save_best_only=True, verbose=0, verbose_description='encoder')
  decoder_checkpointer = utils_vae.CustomCheckpointer(
        save_path_decoder, decoder, monitor=monitor, mode=monitor_mode,
        save_best_only=True, verbose=0, verbose_description='decoder')
  reduce_lr = ReduceLROnPlateau(factor=1/math.sqrt(10), verbose=1,
                                patience=hyperpars['reduce_lr_patience'],
                                min_lr=hyperpars['initial_lr']/100,
                                monitor=monitor,
                                mode=monitor_mode)
  loss_plotter = utils_vae.PlotLosses()
  callbacks = [earlystopper, encoder_checkpointer, decoder_checkpointer,
               reduce_lr, loss_plotter]
  model.fit_generator(train_gen,
                      steps_per_epoch=train_features.shape[0] // (
                          hyperpars['batch_size']),
                      epochs=hyperpars['num_epochs'],
                      callbacks=callbacks,
                      )
  
if 'inspect' in mode:
  K.clear_session()
  encoder = load_model(save_path_encoder)
  decoder = load_model(save_path_decoder)
  utils_vae.plot_results((encoder, decoder),
                         (train_features, test_features, train_eq_ids,
                          train_ttf),
                         hyperpars,
                         model_folder=save_folder)