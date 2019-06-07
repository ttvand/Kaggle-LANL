import datetime
import warnings
with warnings.catch_warnings():
  warnings.simplefilter('ignore', RuntimeWarning)
  import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import preprocess
import random
from scipy.signal import savgol_filter

# Set global seed in order to obtain reproducible results
GLOBAL_SEED = 14
random.seed(GLOBAL_SEED)
np.random.seed(seed=GLOBAL_SEED)

DATA_FOLDER = '/media/tom/cbd_drive/Kaggle/LANL/Data/'
FIGURES_FOLDER = DATA_FOLDER + 'lightGBM_pred_figures'


# Helper function for naming the model path given the fold id
def get_fold_description(fold, num_folds):
  return fold if fold < num_folds else 'all_train'

# Fit a selected model and fold
def fit_model(save_path_orig, fold, num_folds, train_val_split, hyperpars,
              overwrite_train, features, other_features, targets,
              target_quantile):
  fold_description = get_fold_description(fold, num_folds)
  save_path = '{}-{}.txt'.format(save_path_orig, fold_description)
  if not os.path.exists(save_path) or overwrite_train:
    if fold == num_folds:
      train_ids = np.arange(features.shape[0])
    else:
      train_ids = train_val_split[fold][0]
    x_train = features.iloc[train_ids]
    x_weights = other_features.train_weight.values[train_ids]
    y_train = targets[train_ids]
    
    # sklearn interface
    model = lgb.LGBMRegressor(**hyperpars)
    model.fit(x_train, y_train, sample_weight=x_weights)
    model.booster_.save_model(save_path)
    
  # Evaluate OOF performance
  fold_mae = get_fold_mae(save_path_orig, fold, num_folds, train_val_split,
                          features, targets, target_quantile)
  
  return fold_mae
  

# Train all folds
def train_models(save_path, hyperpars, overwrite_train, train_on_all_data,
                 remove_overlap_chunks, train_all_previous, target_quantile,
                 train_last_six_complete=False):
  (features, other_features, targets) = preprocess.get_preprocessed(
      'train', remove_overlap_chunks, scale=NORMALIZE_FEATURES,
      remove_incomplete_eqs=REMOVE_INCOMPLETE_EQS,
      target_quantile=target_quantile,
      train_last_six_complete=train_last_six_complete)
  
  train_val_split = preprocess.train_val_split(
      remove_overlap_chunks, ordered=True, num_folds=NUM_FOLDS,
      remove_incomplete_eqs=REMOVE_INCOMPLETE_EQS,
      train_all_previous=train_all_previous,
      target_quantile=target_quantile,
      train_last_six_complete=train_last_six_complete)
  num_folds = len(train_val_split)
  num_train_models = num_folds + int(train_on_all_data)
  for fold in range(num_train_models):
    print('\nProcessing fold {} of {}'.format(fold+1, num_train_models))
    fit_model(save_path, fold, num_folds, train_val_split, hyperpars,
              overwrite_train, features, other_features, targets,
              target_quantile)
    
# Obtain the OOF validation score for a given model
def get_fold_mae(save_path_orig, fold, num_folds, train_val_split, features,
                 targets, target_quantile, plot_error=True):
  fold_description = get_fold_description(fold, num_folds)
  save_path = '{}-{}.txt'.format(save_path_orig, fold_description)
  loaded_model = lgb.Booster(model_file=save_path)
  valid_ids = train_val_split[fold][1] if fold < num_folds else (
      np.arange(features.shape[0]))
  x_valid = features.iloc[valid_ids]
  y_valid_orig = targets[valid_ids]
  if x_valid.size:
    valid_preds_orig = loaded_model.predict(x_valid)
    if target_quantile:
      valid_preds = train_val_split[fold][7] * (1-valid_preds_orig)
      y_valid = train_val_split[fold][8]
    else:
      valid_preds = valid_preds_orig
      y_valid = y_valid_orig
    oof_mae = np.abs(valid_preds - y_valid).mean()
    
    # Don't normalize the mean when using the quantile method
    if target_quantile:
      valid_multiplier = 1
    else:
      mean_offset = train_val_split[fold][6] - valid_preds.mean()
      valid_multiplier = 1 + mean_offset/valid_preds.mean()
#    oof_mae_norm = np.abs(valid_preds+mean_offset- y_valid).mean()
    oof_mae_norm = np.abs(valid_preds*valid_multiplier - y_valid).mean()
  else:
    oof_mae = 0
    oof_mae_norm = 0
  error_description = "OOF MAE" if fold < num_folds else "Train error"
  print('{}: {}'.format(error_description, np.round(oof_mae, 3)))
  print('{}: {}'.format(error_description+" NORM", np.round(oof_mae_norm, 3)))
  
  # Plot the out of fold predictions versus the ground truth
  if plot_error and x_valid.size:
    quant_ext = '_quantile' if target_quantile else '_ttf'
    filename = os.path.join(
        FIGURES_FOLDER, "oof_preds_" + str(fold) + quant_ext + ".png")
    ordered_ids = valid_ids.argsort()
    valid_ids_ordered = valid_ids[ordered_ids]
    valid_preds_ordered = valid_preds[ordered_ids]
    valid_preds_ordered_sg = savgol_filter(valid_preds_ordered, 51, 3)
    y_valid_ordered = y_valid[ordered_ids]
    plt.figure(figsize=(10, 8))
    plt.plot(valid_ids_ordered, valid_preds_ordered_sg)
    plt.plot(valid_ids_ordered, y_valid_ordered, color='green')
    plt.hlines(valid_preds.min(), valid_ids.min(), valid_ids.max(),
               color='red', linestyles="dashed")
    plt.hlines(valid_preds.max(), valid_ids.min(), valid_ids.max(),
               color='red', linestyles="dashed")
    plt.savefig(filename)
    plt.show()
    
  return oof_mae, oof_mae_norm, valid_ids.size

# Validate all folds
def validate_models(save_path, remove_overlap_chunks, train_all_previous,
                    target_quantile, train_last_six_complete=False):
  (features, other_features, targets) = preprocess.get_preprocessed(
      'train', remove_overlap_chunks, scale=NORMALIZE_FEATURES,
      remove_incomplete_eqs=REMOVE_INCOMPLETE_EQS,
      target_quantile=target_quantile)
  
  train_val_split = preprocess.train_val_split(
      remove_overlap_chunks, ordered=True, num_folds=NUM_FOLDS,
      remove_incomplete_eqs=REMOVE_INCOMPLETE_EQS,
      train_all_previous=train_all_previous,
      target_quantile=target_quantile)
  
  if target_quantile:
    targets = other_features.target_original.values
  num_folds = len(train_val_split)
  sum_maes = []
  maes = []
  fold_mae_norms = []
  total_count = 0
  for fold in range(num_folds):
    print('\nProcessing fold {} of {}'.format(fold+1, num_folds))
    fold_mae, fold_mae_norm, oof_count = get_fold_mae(
        save_path, fold, num_folds, train_val_split, features, targets,
        target_quantile)
    sum_maes.append(fold_mae*oof_count)
    maes.append((fold_mae, oof_count))
    fold_mae_norms.append(fold_mae_norm)
    total_count += oof_count
  av_mae_norm = np.array([n*c for (n, c) in zip(
      fold_mae_norms, [c for (m, c) in maes])]).sum()/total_count
  mae = np.array(sum_maes).sum()/total_count
  maes = [mae] + maes
  print('\nAverage OOF MAE: {}'.format(np.round(mae, 3)))
  print('Average OOF MAE normalized: {}'.format(np.round(av_mae_norm, 3)))
  return maes, av_mae_norm
    
# Test prediction generation - only consider last fold
def test_model(save_path, test_on_all_folds, train_all_previous,
               target_quantile, median_test_cyle_length, seed_ext=None,
               train_last_six_complete=False, drop_first_test_fold=False):
  (x_test, other_test, _) = preprocess.get_preprocessed(
      'test', remove_overlap_chunks=True, scale=NORMALIZE_FEATURES,
      remove_incomplete_eqs=REMOVE_INCOMPLETE_EQS,
      target_quantile=target_quantile)
  train_val_split = preprocess.train_val_split(
      ordered=True, remove_overlap_chunks=True, num_folds=NUM_FOLDS,
      remove_incomplete_eqs=REMOVE_INCOMPLETE_EQS,
      train_all_previous=train_all_previous,
      target_quantile=target_quantile)
  
  num_folds = len(train_val_split)
  num_folds = 1 if train_last_six_complete else num_folds
  pred_folds = [f for f in range(num_folds)] if test_on_all_folds else [
      num_folds-1]
  model_preds = np.zeros((x_test.shape[0], len(pred_folds)))
  for (i, fold) in enumerate(pred_folds):
    print("Making test predictions {} of {}".format(i+1,
          len(pred_folds)))
    fold_description = get_fold_description(fold, num_folds)
    model_path = '{}-{}.txt'.format(save_path, fold_description)
    model = lgb.Booster(model_file=model_path)
    model_preds[:, i] = model.predict(x_test)
  model_preds = model_preds[:, 1:] if drop_first_test_fold else model_preds
  preds_test = np.mean(model_preds, 1)
  
  if target_quantile:
    preds_test = median_test_cyle_length*(1-preds_test)
  
  # Write the output pandas data frame
  submission = pd.read_csv(DATA_FOLDER + 'sample_submission.csv')
  submission.time_to_failure = preds_test
  the_date = datetime.datetime.now().strftime('%y-%m-%d-%H-%M')
  the_date = the_date if seed_ext is None else the_date + seed_ext
  submission_path = '/home/tom/Kaggle/LANL/Submissions/' + the_date + '.csv'
  submission.to_csv(submission_path, index=False)
  

###################################################
# Main logic for training, validation and testing #
###################################################
mode = ['train', 'validation', 'train_validate', 'test', 'no_action',
        'final_test'][5]
model_description = 'sequential_lgbm'
overwrite_train = True
train_on_all_data = False
NORMALIZE_FEATURES = True
REMOVE_INCOMPLETE_EQS = False # Delete train_val_split.pickle after changing
train_all_previous = True
test_on_all_folds = True
median_test_cyle_length = [8, 9.7, 11.0, 12.0][3] # public LB and private LB values
remove_overlap_chunks = False
target_quantile = True
train_test_num_eqs = [3, 4][0] # Include cycle 13 or not? 4 means included
NUM_FOLDS = 6 if train_test_num_eqs == 3 else 5 # Delete train_val_split.pickle after changing
num_final_test_seeds = 10

hyperpars = {
    'objective': 'mae',
    'boosting': 'gbdt',
    'verbosity': -1,
    'random_seed': GLOBAL_SEED,
    'n_estimators': 3000,
    
    'learning_rate': [0.005, 0.01, 0.02][1],
    'num_leaves': list(range(10, 24, 2))[3],
    'max_depth': [8, 12, 16, -1][1],
    'feature_fraction': [0.001, 0.003, 0.01][1],
    'subsample': [0.4, 0.6, 0.8, 1.0][1],
    'lambda_l1': [0, 0.1, 0.2, 0.4, 0.6, 0.9][3],
    'lambda_l2': [0, 0.1, 0.2, 0.4, 0.6, 0.9][3],
    'min_data_in_leaf': [10, 20, 40, 60, 80, 100][3],
    'min_gain_to_split': [0, 0.001, 0.01, 0.1][1],
}

save_path = '/home/tom/Kaggle/LANL/Models/' + model_description
if mode == 'train':
  train_models(save_path, hyperpars, overwrite_train, train_on_all_data,
               remove_overlap_chunks, train_all_previous, target_quantile)
elif mode == 'validation':
  valid_maes = validate_models(save_path, remove_overlap_chunks,
                               train_all_previous, target_quantile)
elif mode == 'train_validate':
  train_models(save_path, hyperpars, overwrite_train, train_on_all_data,
               remove_overlap_chunks, train_all_previous, target_quantile)
  print("\n")
  validate_models(save_path, remove_overlap_chunks, train_all_previous,
                  target_quantile)
elif mode == 'test':
  test_model(save_path, test_on_all_folds, train_all_previous, target_quantile,
             median_test_cyle_length)
elif mode == 'final_test':
  # Models trained on the last 6 complete cycles
  for i in range(num_final_test_seeds):
    print('\nFinal test predictions {} of {}'.format(
        i+1, num_final_test_seeds))
    hyperpars['random_seed'] = i+1
    NUM_FOLDS = 1
    train_models(save_path, hyperpars, overwrite_train=True,
                 train_on_all_data=False, remove_overlap_chunks=False,
                 train_all_previous=False, target_quantile=True,
                 train_last_six_complete=True)
    
    test_model(save_path, test_on_all_folds=False, train_all_previous=False,
               target_quantile=True,
               median_test_cyle_length=median_test_cyle_length,
               seed_ext=' - lgbm_last_six_seed_' + str(i+1) + '_of_' + str(
                   num_final_test_seeds) + '_' + str(median_test_cyle_length),
                   train_last_six_complete=True)
  
  # Models trained on all but the first folds using all data up to that point
  for i in range(num_final_test_seeds):
    print('\nFinal test predictions {} of {}'.format(
        i+1, num_final_test_seeds))
    hyperpars['random_seed'] = i+1
    NUM_FOLDS = 6 if train_test_num_eqs == 3 else 5
    train_models(save_path, hyperpars, overwrite_train=True,
                 train_on_all_data=False, remove_overlap_chunks=False,
                 train_all_previous=True, target_quantile=True,
                 train_last_six_complete=False)
    
    test_model(save_path, test_on_all_folds=True, train_all_previous=True,
               target_quantile=True,
               median_test_cyle_length=median_test_cyle_length,
               seed_ext=' - lgbm_all_prev_seed_' + str(i+1) + '_of_' + str(
                   num_final_test_seeds) + '_' + str(median_test_cyle_length),
                   train_last_six_complete=False, drop_first_test_fold=True)