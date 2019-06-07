import datetime
import warnings
with warnings.catch_warnings():
  warnings.simplefilter('ignore', RuntimeWarning)
  import lightgbm as lgb
import numpy as np
import os
import pandas as pd
import preprocess
import random

# Set global seed in order to obtain reproducible results
GLOBAL_SEED = 14
random.seed(GLOBAL_SEED)
np.random.seed(seed=GLOBAL_SEED)


# Helper function for naming the model path given the fold id
def get_fold_description(fold, num_folds):
  return fold if fold < num_folds else 'all_train'

# Fit a selected model and fold
def fit_model(save_path_orig, split, fold, num_folds, train_val_split,
              hyperpars, overwrite_train, features, other_features, targets,
              early_stopping):
  fold_description = get_fold_description(fold, num_folds)
  save_path = '{}-{}-{}.txt'.format(save_path_orig, split, fold_description)
  if not os.path.exists(save_path) or overwrite_train:
    if fold == num_folds:
      train_ids = np.arange(features.shape[0])
    else:
      train_ids = train_val_split[split][fold][2 if early_stopping else 0]
    x_train = features.iloc[train_ids]
    x_weights = other_features.train_weight.values[train_ids]
    y_train = targets[train_ids]
    
    # sklearn interface  
    model = lgb.LGBMRegressor(**hyperpars)
    if early_stopping and fold < num_folds:
      model.n_estimators *= 5
      es_ids = train_val_split[split][fold][3]
      model.fit(x_train, y_train, sample_weight=x_weights,
                early_stopping_rounds=100, verbose=False,
                eval_set=(features.iloc[es_ids], targets[es_ids]))
    else:
      model.fit(x_train, y_train, sample_weight=x_weights)
    model.booster_.save_model(save_path)
    
  # Evaluate OOF performance
  fold_mae = get_fold_mae(save_path_orig, split, fold, num_folds,
                          train_val_split, features, targets)
  
  return fold_mae
  

# Train all folds and random splits
def train_models(save_path, splits, hyperpars, overwrite_train,
                 early_stopping, train_on_all_data, remove_overlap_chunks):
  (features, other_features, targets) = preprocess.get_preprocessed(
      'train', remove_overlap_chunks, scale=NORMALIZE_FEATURES)
  
  train_val_split = preprocess.train_val_split(remove_overlap_chunks)
  num_folds = len(train_val_split[0])
  num_train_models = num_folds + int(train_on_all_data)
  for split_id, split in enumerate(splits):
    print('Processing split {} of {}'.format(split_id+1, len(splits)))
    for fold in range(num_train_models):
      print('\nProcessing fold {} of {}'.format(fold+1, num_train_models))
      fit_model(save_path, split, fold, num_folds, train_val_split, hyperpars,
                overwrite_train, features, other_features, targets,
                early_stopping)
      
# Obtain the OOF validation score for a given model
def get_fold_mae(save_path_orig, split, fold, num_folds, train_val_split,
                 features, targets):
  fold_description = get_fold_description(fold, num_folds)
  save_path = '{}-{}-{}.txt'.format(save_path_orig, split, fold_description)
  loaded_model = lgb.Booster(model_file=save_path)
  valid_ids = train_val_split[split][fold][1] if fold < num_folds else (
      np.arange(features.shape[0]))
  x_valid = features.iloc[valid_ids]
  y_valid = targets[valid_ids]
  valid_preds = loaded_model.predict(x_valid)
  oof_mae = np.abs(valid_preds - y_valid).mean()
  error_description = "OOF MAE" if fold < num_folds else "Train error"
  print('{}: {}'.format(error_description, np.round(oof_mae, 3)))
  
  return oof_mae, valid_ids.size

# Validate all folds and random splits
def validate_models(save_path, splits, remove_overlap_chunks):
  (features, other_features, targets) = preprocess.get_preprocessed(
      'train', remove_overlap_chunks, scale=NORMALIZE_FEATURES)
  
  train_val_split = preprocess.train_val_split(remove_overlap_chunks)
  num_folds = len(train_val_split[0])
  valid_maes = []
  for split_id, split in enumerate(splits):
    print('Processing split {} of {}'.format(split_id+1, len(splits)))
    sum_split_maes = []
    split_maes = []
    for fold in range(num_folds):
      print('Processing fold {} of {}'.format(fold+1, num_folds))
      fold_mae, oof_count = get_fold_mae(
          save_path, split, fold, num_folds, train_val_split, features,
          targets)
      sum_split_maes.append(fold_mae*oof_count)
      split_maes.append((fold_mae, oof_count))
    split_mae = np.array(sum_split_maes).sum()/features.shape[0]
    split_maes = [split_mae] + split_maes
    print('Split OOF MAE: {}'.format(np.round(split_mae, 3)))
    valid_maes.append((split_maes, splits[split_id]))
  return valid_maes
    
# Test prediction generation - only consider first split for now
def test_model(save_path, split, test_on_all_data):
  (x_test, other_test, _) = preprocess.get_preprocessed(
      'test', remove_overlap_chunks=True, scale=NORMALIZE_FEATURES)
  train_val_split = preprocess.train_val_split(remove_overlap_chunks=True)
  num_folds = len(train_val_split[split])
  num_test = x_test.shape[0]
  num_prediction_models = num_folds + int(test_on_all_data)
  model_preds = np.zeros((num_test, num_prediction_models))
  for fold in range(num_prediction_models):
    print("Making test predictions {} of {}".format(fold+1,
          num_prediction_models))
    fold_description = get_fold_description(fold, num_folds)
    model_path = '{}-{}-{}.txt'.format(save_path, split, fold_description)
    model = lgb.Booster(model_file=model_path)
    model_preds[:, fold] = model.predict(x_test)
  
  # Write the output pandas data frame
  preds_test = np.mean(model_preds, 1)
  submission = pd.read_csv('/home/tom/Kaggle/LANL/Data/sample_submission.csv')
  submission.time_to_failure = preds_test
  the_date = datetime.datetime.now().strftime('%y-%m-%d-%H-%M')
  submission_path = '/home/tom/Kaggle/LANL/Submissions/' + the_date + '.csv'
  submission.to_csv(submission_path, index=False)
  

###################################################
# Main logic for training, validation and testing #
###################################################
mode = ['train', 'validation', 'train_validate', 'test', 'no_action'][0]
splits = [6]
model_description = 'initial_lgbm'
overwrite_train = True
early_stopping = False # Better results without leave one EQ out ES
train_on_all_data = True
test_on_all_data = train_on_all_data
NORMALIZE_FEATURES = True
remove_overlap_chunks = True

hyperpars = {
    'objective': 'regression_l1',
    'boosting': 'gbdt',
    'verbosity': -1,
    'random_seed': GLOBAL_SEED,
    'n_estimators': 1000,
    
    'learning_rate': [0.0025, 0.005, 0.01][1],
    'num_leaves': list(range(8, 20, 2))[3],
    'max_depth': [6, 8, 12, 16, -1][2],
    'feature_fraction': [0.8, 0.9, 1][1],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1][2],
    'lambda_l1': [0, 0.1, 0.2, 0.4, 0.6, 0.9][3],
    'lambda_l2': [0, 0.1, 0.2, 0.4, 0.6, 0.9][3],
    'min_data_in_leaf': [10, 20, 40, 60, 100][2],
    'min_gain_to_split': [0, 0.001, 0.01, 0.1][1],
}

save_path = '/home/tom/Kaggle/LANL/Models/' + model_description
if mode == 'train':
  train_models(save_path, splits, hyperpars, overwrite_train, early_stopping,
               train_on_all_data, remove_overlap_chunks)
elif mode == 'validation':
  valid_maes = validate_models(save_path, splits, remove_overlap_chunks)
elif mode == 'train_validate':
  train_models(save_path, splits, hyperpars, overwrite_train, early_stopping,
               train_on_all_data, remove_overlap_chunks)
  print("\n")
  validate_models(save_path, splits, remove_overlap_chunks)
elif mode == 'test':
  test_model(save_path, splits[0], test_on_all_data)
