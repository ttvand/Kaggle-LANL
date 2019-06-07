import warnings
with warnings.catch_warnings():
  warnings.simplefilter('ignore', RuntimeWarning)
  import models
from collections import OrderedDict
import datetime
import gc
import pandas as pd
import train_valid_test_gap

data_folder = '/media/tom/cbd_drive/Kaggle/LANL/Data/'
gc.collect()
splits = [0]
model_description = 'initial_gap'
custom_model = models.initial_gap

# Sweeped parameter settings
param_settings = [
    {
        'block_steps': 4000,
        'epochs': 100,
        'steps_per_epoch': 2000,
        'max_gap_shifts': 1,
        'initial_lr': 1e-3,
        'batch_size': 32,
        'es_patience': 20,
        'reduce_lr_patience': 3,
        'drop_extreme_part_loss_frac': 0.1,
        'recurrent_cells': [32, 32, 32],
        'prediction_layers': [64, 32, 10],
        'gru_dropout': 0.25,
        'prediction_dropout': 0.25,
    },
#        {
#        'block_steps': 2000,
#        'epochs': 100,
#        'steps_per_epoch': 2000,
#        'max_gap_shifts': 1,
#        'initial_lr': 1e-3,
#        'batch_size': 32,
#        'es_patience': 20,
#        'reduce_lr_patience': 3,
#        'drop_extreme_part_loss_frac': 0.1,
#        'recurrent_cells': [32, 32, 32],
#        'prediction_layers': [64, 32, 10],
#        'gru_dropout': 0.25,
#        'prediction_dropout': 0.25,
#    }
]

model_path_base = '/home/tom/Kaggle/LANL/Models/' + model_description
the_date = datetime.datetime.now().strftime('%y-%m-%d-%H-%M')
sweep_path = data_folder + 'hyperpar_sweep_gaps_' + the_date + (
    '.csv')
sweep_summaries = []
for i in range(len(param_settings)):
  print('Hyperpar setting {} of {}'.format(i+1, len(param_settings)))
  hyperpars = param_settings[i]
  selected_grid = OrderedDict(sorted(hyperpars.items()))
  model_path = model_path_base + '_setting_' + str(i+1)
  train_valid_test_gap.train_models(
      custom_model, model_path, splits, hyperpars, overwrite_train=True,
      model_on_all_data=False, max_fold=1)
  valid_ratios = train_valid_test_gap.validate_models(
      model_path, splits, max_fold=1)
  cv = OrderedDict()
  cv['Ratio'] = valid_ratios[0][0][0]
  summary_dict = OrderedDict(list(cv.items()) + list(selected_grid.items()))
  summary_dict['split'] = valid_ratios[0][1]
  sweep_summaries.append(summary_dict)
  
  sweep_results = pd.DataFrame(sweep_summaries)
  sweep_results.to_csv(sweep_path, index=False)
