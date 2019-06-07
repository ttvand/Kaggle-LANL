from collections import OrderedDict
import datetime
import pandas as pd
import random
import train_valid_test_lightgbm_sequential

num_random_hyperpars = 1000
model_description = 'sequential_lgbm'
remove_overlap_chunks = False

# Modeling parameters that relate to the modeled data and target
model_params = {
    'train_all_previous': [False, True],
    'target_quantile': [True],
    }

fixed_params = {
    'objective': 'mae',
    'boosting': 'gbdt',
    'verbosity': -1,
    'random_seed': train_valid_test_lightgbm_sequential.GLOBAL_SEED,
    'usemissing': False,
    }

# In order of apparent effect on OOF performance
param_grid = {
    'n_estimators': [2000, 3000, 5000],
    'learning_rate': [0.005, 0.01, 0.02],
    'num_leaves': list(range(10, 24, 2)),
    
    'feature_fraction': [0.001, 0.003, 0.01],
    'max_depth': [8, 12, 16, -1],
    'min_data_in_leaf': [10, 20, 40, 60, 80, 100],
    'subsample': [0.4, 0.6, 0.8, 1.0],
    'lambda_l1': [0, 0.1, 0.2, 0.4, 0.6, 0.9],
    'lambda_l2': [0, 0.1, 0.2, 0.4, 0.6, 0.9],
    'min_gain_to_split': [0, 0.001, 0.01, 0.1],
}

model_path = '/home/tom/Kaggle/LANL/Models/' + model_description
the_date = datetime.datetime.now().strftime('%y-%m-%d-%H-%M')
sweep_path = '/home/tom/Kaggle/LANL/Data/hyperpar_sweep_lgb_sequential' + (
    the_date + '.csv')
sweep_summaries = []
for i in range(num_random_hyperpars):
  print('\nRandom hyperpar setting {} of {}'.format(i+1, num_random_hyperpars))
  hyperpars = {k: random.choice(v) for k, v in param_grid.items()}
  train_all_previous = random.choice(model_params['train_all_previous'])
  target_quantile = random.choice(model_params['target_quantile'])
  selected_grid = OrderedDict(sorted(hyperpars.items()))
  hyperpars.update(fixed_params)
  train_valid_test_lightgbm_sequential.train_models(
      model_path, hyperpars, overwrite_train=True,
      train_on_all_data=False, remove_overlap_chunks=remove_overlap_chunks,
      train_all_previous=train_all_previous, target_quantile=target_quantile)
  valid_maes = train_valid_test_lightgbm_sequential.validate_models(
      model_path, remove_overlap_chunks, train_all_previous, target_quantile)
  cv = OrderedDict()
  cv['MAE'] = valid_maes[0][0]
  cv['MAE_normalized'] = valid_maes[1]
  summary_dict = OrderedDict(list(cv.items()) + list(selected_grid.items()))
  summary_dict['train_all_previous'] = train_all_previous
  summary_dict['target_quantile'] = target_quantile
  summary_dict['remove_overlap_chunks'] = remove_overlap_chunks
  sweep_summaries.append(summary_dict)
  
  sweep_results = pd.DataFrame(sweep_summaries)
  sweep_results.to_csv(sweep_path, index=False)
