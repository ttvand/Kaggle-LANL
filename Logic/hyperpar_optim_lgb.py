from collections import OrderedDict
import datetime
import pandas as pd
import random
import train_valid_test_lightgbm

num_random_hyperpars = 1000
splits = [0]
model_description = 'initial_lgbm'
early_stopping = False # Consistently better results without leave one EQ ES
remove_overlap_chunks = True

fixed_params = {
    'objective': 'regression_l1',
    'boosting': 'gbdt',
    'verbosity': -1,
    'random_seed': train_valid_test_lightgbm.GLOBAL_SEED,
    }

# In order of apparent effect on OOF performance
param_grid = {
    'n_estimators': [700, 1000, 1500],
    'learning_rate': [0.0025, 0.005, 0.01],
    'num_leaves': list(range(8, 20, 2)),
    
    'feature_fraction': [0.8, 0.9, 1],
    'max_depth': [6, 8, 12, 16, -1],
    'min_data_in_leaf': [10, 20, 40, 60, 100],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1],
    'lambda_l1': [0, 0.1, 0.2, 0.4, 0.6, 0.9],
    'lambda_l2': [0, 0.1, 0.2, 0.4, 0.6, 0.9],
    'min_gain_to_split': [0, 0.001, 0.01, 0.1],
}

model_path = '/home/tom/Kaggle/LANL/Models/' + model_description
the_date = datetime.datetime.now().strftime('%y-%m-%d-%H-%M')
sweep_path = '/home/tom/Kaggle/LANL/Data/hyperpar_sweep_lgb_' + the_date + (
    '.csv')
sweep_summaries = []
for i in range(num_random_hyperpars):
  print('Random hyperpar setting {} of {}'.format(i+1, num_random_hyperpars))
  hyperpars = {k: random.choice(v) for k, v in param_grid.items()}
  selected_grid = OrderedDict(sorted(hyperpars.items()))
  hyperpars.update(fixed_params)
  train_valid_test_lightgbm.train_models(
      model_path, splits, hyperpars, overwrite_train=True,
      early_stopping=early_stopping, train_on_all_data=False,
      remove_overlap_chunks=remove_overlap_chunks)
  valid_maes = train_valid_test_lightgbm.validate_models(
      model_path, splits, remove_overlap_chunks)
  cv = OrderedDict()
  cv['MAE'] = valid_maes[0][0][0]
  summary_dict = OrderedDict(list(cv.items()) + list(selected_grid.items()))
  summary_dict['split'] = valid_maes[0][1]
  summary_dict['remove_overlap_chunks'] = remove_overlap_chunks
  sweep_summaries.append(summary_dict)
  
  sweep_results = pd.DataFrame(sweep_summaries)
  sweep_results.to_csv(sweep_path, index=False)
