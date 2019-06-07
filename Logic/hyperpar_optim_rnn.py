import warnings
with warnings.catch_warnings():
  warnings.simplefilter('ignore', RuntimeWarning)
  import models
from collections import OrderedDict
import datetime
import pandas as pd
import random
import train_valid_test_rnn

num_random_hyperpars = 500
splits = [0]
model_description = 'initial_gru'
custom_model = models.initial_gru
remove_overlap_chunks = True

# Sweeped parameters
param_grid = {
    'epochs': [20, 30, 50],
    'validation_split': [0.05, 0.1, 0.2],
    'es_patience': [5, 10, 20],
    'initial_lr': [1e-4, 3e-4, 1e-3, 3e-3],
    'batch_size': [32],
    'reduce_lr_patience': [3, 5, 10, 100],
    'train_last_part_loss_frac': [0.1, 0.2, 0.4],
    'predict_last_part_frac': [0.01, 0.05, 0.1, 0.2, 0.5],
    'encoding_layers': [[], [32], [32, 32], [32, 32, 32],
                        [64], [64, 64], [64, 64, 64],
                        [128], [128, 128], [128, 128, 128]],
    'encoding_input_dropout': [0.2, 0.3, 0.4, 0.5, 0.6],
    'num_recurrent_cells': [32, 64, 128, 256, 512],
    'prediction_layers': [[], [10], [32], [16, 16], [32, 32]],
}

model_path = '/home/tom/Kaggle/LANL/Models/' + model_description
the_date = datetime.datetime.now().strftime('%y-%m-%d-%H-%M')
sweep_path = '/home/tom/Kaggle/LANL/Data/hyperpar_sweep_rnn_' + the_date + (
    '.csv')
sweep_summaries = []
for i in range(num_random_hyperpars):
  print('Random hyperpar setting {} of {}'.format(i+1, num_random_hyperpars))
  hyperpars = {k: random.choice(v) for k, v in param_grid.items()}
  selected_grid = OrderedDict(sorted(hyperpars.items()))
  train_valid_test_rnn.train_models(
      custom_model, model_path, splits, hyperpars, overwrite_train=True,
      model_on_all_data=False, remove_overlap_chunks=remove_overlap_chunks)
  valid_maes = train_valid_test_rnn.validate_models(
      model_path, splits, remove_overlap_chunks=False)
  cv = OrderedDict()
  cv['MAE'] = valid_maes[0][0][0]
  summary_dict = OrderedDict(list(cv.items()) + list(selected_grid.items()))
  summary_dict['split'] = valid_maes[0][1]
  summary_dict['remove_overlap_chunks'] = remove_overlap_chunks
  sweep_summaries.append(summary_dict)
  
  sweep_results = pd.DataFrame(sweep_summaries)
  sweep_results.to_csv(sweep_path, index=False)
