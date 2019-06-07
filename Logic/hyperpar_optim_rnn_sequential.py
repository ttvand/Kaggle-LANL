import warnings
with warnings.catch_warnings():
  warnings.simplefilter('ignore', RuntimeWarning)
  import models
from collections import OrderedDict
import datetime
import os
import pandas as pd
import random
import train_valid_test_rnn_sequential

num_random_hyperpars = 1000
model_description = 'sequential_gru'
custom_model = models.sequential_gru
remove_overlap_chunks = True
skip_last_train_fold = True
continuing_experiment = ['19-06-01-21-48.csv', None][1]

# Modeling parameters that relate to the modeled data and target
model_params = {
    'train_all_previous': [False, True],
    'target_quantile': [True],
    }

fixed_params = {
    'block_steps': 1500,
    'chunk_blocks': 100,
    'input_dimension_freq': 40,
    'input_dimension_cpc': 16,
    'include_cpc_features': False, # This kills test performance! Leakage through these CPC features!
    'include_freq_features': True,
    'epochs': 10,
    'steps_per_epoch': 1000,
    
    'es_patience': 20,
    'batch_size': 32,
    'predict_last_part_frac': 0.1,
    
    'train_valid_batch': 1000,
    'validation_valid_batch': 2000,
    }

# In order of apparent effect on OOF performance
param_grid = {
    'initial_lr': [5e-5, 1e-4, 3e-4],
    'reduce_lr_patience': [1, 3],
    'train_last_part_loss_frac': [0.4, 0.2, 0.1],
    'encoding_layers': [[32], [32, 32]],
    'relu_last_encoding_layer': [True],
    'encoding_input_dropout': [0, 0.15, 0.3],
    'encoding_dropout': [0, 0.1, 0.2, 0.3],
    'recurrent_cells': [[8], [32], [8, 8], [32, 16], [64, 32], [16, 16, 16]],
    'gru_dropout': [0, 0.25],
    'prediction_layers': [[], [8], [32], [8, 8]],
}

model_path = '/home/tom/Kaggle/LANL/Models/' + model_description

the_date = datetime.datetime.now().strftime('%y-%m-%d-%H-%M')
base_sweep_path = '/home/tom/Kaggle/LANL/Data/hyperpar_sweep_rnn_sequential'
sweep_path = base_sweep_path + the_date + '.csv'
sweep_summaries = []
# Optionally, continue an existing experiment
if continuing_experiment is not None:
  continuing_path = base_sweep_path + continuing_experiment
  if os.path.exists(continuing_path):
    sweep_path = continuing_path
    continuing_data = pd.read_csv(continuing_path)
    data_cols = continuing_data.columns
    sweep_summaries = []
    for i in range(continuing_data.shape[0]):
      data_row = continuing_data.iloc[i]
      value_tuple = [(c, data_row[c]) for c in data_cols]
      sweep_summaries.append(OrderedDict(value_tuple))

for i in range(num_random_hyperpars):
  print('Random hyperpar setting {} of {}'.format(i+1, num_random_hyperpars))
  hyperpars = {k: random.choice(v) for k, v in param_grid.items()}
  train_all_previous = random.choice(model_params['train_all_previous'])
  target_quantile = random.choice(model_params['target_quantile'])
  selected_grid = OrderedDict(sorted(hyperpars.items()))
  hyperpars.update(fixed_params)
  hyperpars['clip_preds_zero_one'] = target_quantile
  train_valid_test_rnn_sequential.train_models(
      custom_model, model_path, hyperpars, overwrite_train=True,
      train_on_all_data=False, remove_overlap_chunks=remove_overlap_chunks,
      train_all_previous=train_all_previous,
      skip_last_train_fold=skip_last_train_fold,
      target_quantile=target_quantile)
  valid_maes = train_valid_test_rnn_sequential.validate_models(
      model_path, hyperpars, remove_overlap_chunks, train_all_previous,
      target_quantile)
  cv = OrderedDict()
  cv['MAE'] = valid_maes[0][0]
  cv['MAE_normalized'] = valid_maes[1]
  summary_dict = OrderedDict(list(cv.items()) + list(selected_grid.items()))
  summary_dict['train_all_previous'] = train_all_previous
  summary_dict['target_quantile'] = target_quantile
  sweep_summaries.append(summary_dict)
  
  sweep_results = pd.DataFrame(sweep_summaries)
  sweep_results.to_csv(sweep_path, index=False)
