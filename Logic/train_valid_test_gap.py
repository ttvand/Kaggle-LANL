import warnings
with warnings.catch_warnings():
  warnings.simplefilter('ignore', RuntimeWarning)
  import models
import math
import numpy as np
import os
import pandas as pd
import preprocess
import utils

from keras import backend as K
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau

# Set the path to the data folder
data_folder = '/media/tom/cbd_drive/Kaggle/LANL/Data/'
#data_folder = '/home/tom/Kaggle/LANL/Data/'

# Set global seed in order to obtain reproducible results
GLOBAL_SEED = 14
utils.make_results_reproducible(K, GLOBAL_SEED)

# Load the global variable GAP_DATA if it has not been loaded before
if (not 'GAP_DATA' in locals()) and (not 'GAP_DATA' in globals()):
  GAP_DATA = pd.read_csv(data_folder + 'gap_data.csv')

# Helper function for naming the model path given the fold id
def get_fold_description(fold, num_folds):
  return fold if fold < num_folds else 'all_train'

# Fit a selected model and fold
def fit_model(custom_model, save_path_orig, split, fold, num_folds,
              train_val_split, hyperpars, overwrite_train):
  fold_description = get_fold_description(fold, num_folds)
  save_path = '{}-{}-{}.h5'.format(save_path_orig, split, fold_description)
  if not os.path.exists(save_path) or overwrite_train:
    if fold == num_folds:
      train_ranges = (np.array([0]), np.array([GAP_DATA.shape[0]]))
    else:
      train_ranges = train_val_split[split][fold][0]
    train_gen = utils.generator_gap_batch(GAP_DATA, train_ranges, hyperpars)
    
    (inputs, outputs) = custom_model(hyperpars)
    model = Model(inputs=[inputs], outputs=[outputs])
    adam = Adam(lr=hyperpars['initial_lr'])
    train_loss = utils.wrapped_partial(
        utils.xe_central_part_loss, drop_extreme_part=(
        hyperpars['drop_extreme_part_loss_frac']))
    ratio_metric = utils.wrapped_partial(
        utils.gap_predict_ratio, drop_extreme_part=(
        hyperpars['drop_extreme_part_loss_frac']))
    model.compile(optimizer=adam, loss=train_loss,
                  metrics=[ratio_metric])
#    (validation_monitor, validation_mode) = ('gap_predict_ratio', 'max')
    (validation_monitor, validation_mode) = ('loss', 'min')
    earlystopper = EarlyStopping(
        monitor=validation_monitor, mode=validation_mode,
        patience=hyperpars['es_patience'], verbose=1)
    checkpointer = ModelCheckpoint(save_path, monitor=validation_monitor,
                                   mode=validation_mode, verbose=1,
                                   save_best_only=True, period=1)
    reduce_lr = ReduceLROnPlateau(factor=1/math.sqrt(10), verbose=1,
                                  patience=hyperpars['reduce_lr_patience'],
                                  min_lr=hyperpars['initial_lr']/100,
                                  monitor=validation_monitor,
                                  mode=validation_mode)
    callbacks = [earlystopper, checkpointer, reduce_lr]
    callbacks.append(utils.PlotLosses())
#    callbacks.append(TensorBoard('./Graph'))
    
    model.fit_generator(train_gen,
                        steps_per_epoch=hyperpars['steps_per_epoch'],
                        epochs=hyperpars['epochs'],
                        callbacks=callbacks,
                        validation_data=train_gen,
                        validation_steps=1,
                        )
    
  # Evaluate OOF performance
  get_fold_ratio(save_path_orig, split, fold, num_folds, train_val_split,
                 hyperpars)
  

# Train all folds and random splits
def train_models(custom_model, save_path, splits, hyperpars, overwrite_train,
                 model_on_all_data, max_fold):
  train_val_split = preprocess.train_val_split_gaps()
  num_folds = len(train_val_split[0])
  num_train_models = min(max_fold, num_folds + int(model_on_all_data))
  for split_id, split in enumerate(splits):
    print('Processing split {} of {}'.format(split_id+1, len(splits)))
    for fold in range(num_train_models):
      print('\nProcessing fold {} of {}'.format(fold+1, num_train_models))
      K.clear_session()
      fit_model(custom_model, save_path, split, fold, num_folds,
                train_val_split, hyperpars, overwrite_train)
      
      
# Obtain the OOF validation score for a given model
def get_fold_ratio(save_path_orig, split, fold, num_folds, train_val_split,
                   hyperpars):
  valid_ranges = train_val_split[split][fold][1] if fold < num_folds else (
      (np.array([0]), np.array([GAP_DATA.shape[0]])))
  if 'num_validation_blocks' in hyperpars.keys():
    max_valid_rows = hyperpars['num_validation_blocks']
  else:
    max_valid_rows = float('inf')
  (x_valid, y_valid, _) = utils.get_gap_prediction_features(
      GAP_DATA, valid_ranges, hyperpars, order_start_rows=False,
      max_considered_start_rows=max_valid_rows)
  valid_preds = make_predictions(save_path_orig, split, fold, num_folds,
                                 x_valid)
  gap_av_pred, no_gap_av_preds = utils.gap_pred_statistics(
      y_valid, valid_preds, hyperpars['drop_extreme_part_loss_frac'])
#  gap_preds_ratio = np.reshape(valid_preds.flatten()[
#      y_valid.flatten().astype(np.bool)]/valid_preds.mean(), [-1, 16])
  error_description = "OOF gap pred" if fold < num_folds else "Train error"
  print('{}: {}'.format(error_description, (
      np.round(gap_av_pred, 5), np.round(no_gap_av_preds, 5))))
  
  return gap_av_pred/no_gap_av_preds, x_valid.shape[0]


# Helper function for generating model predictions
def make_predictions(save_path_orig, split, fold, num_folds, x_features):
  fold_description = get_fold_description(fold, num_folds)
  save_path = '{}-{}-{}.h5'.format(save_path_orig, split, fold_description)
  with warnings.catch_warnings():
    warnings.simplefilter('ignore', RuntimeWarning)
    model = load_model(save_path, custom_objects={
            'xe_central_part_loss': utils.xe_central_part_loss,
            'gap_predict_ratio': utils.gap_predict_ratio})
  preds = model.predict(x_features, verbose=1)
  return preds


# Validate all folds and random splits
def validate_models(save_path, splits, max_fold)  :
  train_val_split = preprocess.train_val_split_gaps()
  num_folds = min(max_fold, len(train_val_split[0]))
  valid_ratios = []
  for split_id, split in enumerate(splits):
    print('Processing split {} of {}'.format(split_id+1, len(splits)))
    sum_split_ratios = []
    split_ratios = []
    oof_counts = []
    for fold in range(num_folds):
      print('Processing fold {} of {}'.format(fold+1, num_folds))
      fold_ratio, oof_count = get_fold_ratio(
          save_path, split, fold, num_folds, train_val_split, hyperpars)
      sum_split_ratios.append(fold_ratio*oof_count)
      split_ratios.append((fold_ratio, oof_count))
      oof_counts.append(oof_count)
    split_ratio = np.array(sum_split_ratios).sum()/np.array(oof_counts).sum()
    split_ratios = [split_ratio] + split_ratios 
    print('Split OOF ratio: {}'.format(np.round(split_ratio, 3)))
    valid_ratios.append((split_ratios, splits[split_id]))
  return valid_ratios


# Generate and store gap test predictions for all chunks
def test_model(model_path, save_path, split, hyperpars):
  x_test = pd.read_csv(data_folder + 'test_combined.csv')
  x_test['is_gap'] = True
#  x_test = x_test[:600000] # For verifying the logic quickly
  test_file_steps = 150000
  num_test_files = int(x_test.shape[0]/test_file_steps)
  test_ranges = (test_file_steps*np.arange(num_test_files),
                 test_file_steps*(1+np.arange(num_test_files))-(
                     hyperpars['block_steps']))
  (x_test_batched, _, test_start_rows) = utils.get_gap_prediction_features(
      x_test, test_ranges, hyperpars, order_start_rows=True)
  test_preds = make_predictions(model_path, split, fold=0, num_folds=1,
                                x_features=x_test_batched)
  test_gap_preds_aligned = utils.align_test_gap_preds(
      test_preds, test_file_steps, test_start_rows, hyperpars)
  data_path = save_path + '_aligned_predictions_test' + '.csv'
  test_gap_preds_aligned.to_csv(data_path, index=False)
  

# Generate and store gap validation predictions for the first 11 validation eqs
# The resulting gap predictions are of approximately the same dimensions as
# the test gap probability predictions.
def validate_save_gap_preds(model_path, save_path, split, hyperpars):
  # Determine the first eleven validation earthquake ids
  num_first_eqs = 11
  train_val_split = preprocess.train_val_split_gaps()
  first_val_ranges = train_val_split[split][0][1] # First fold, validation
  other_train_features = preprocess.get_preprocessed(
      'train', remove_overlap_chunks=True)[1]
  first_eq_id_other_features = np.where(
      other_train_features.start_row.values == first_val_ranges[0][0])[0][0]
  first_eq_id = other_train_features.eq_id.values[first_eq_id_other_features]
  first_row_next_eq = other_train_features.start_row.values[np.where(
      other_train_features.eq_id.values == first_eq_id+num_first_eqs)[0][0]]
  first_valid_eq_ids = np.arange(first_val_ranges[0][0], first_row_next_eq)
  
  # Drop the last part of the valid_eq_ids that don't contain an entire chunk
  valid_file_steps = 150000
  new_eq_ids = np.where(np.diff(other_train_features.eq_id.values) > 0)[0] + 1
  drop_eq_end_ids = new_eq_ids[new_eq_ids > first_eq_id_other_features][
      :num_first_eqs]
  drop_ids = np.array([])
  for i in range(num_first_eqs):
    drop_ids_eq = np.arange(
        other_train_features.start_row.values[
            drop_eq_end_ids[i]-2]+valid_file_steps,
        other_train_features.start_row.values[drop_eq_end_ids[i]])
    drop_ids = np.append(drop_ids, drop_ids_eq)
    
  first_valid_eq_ids = np.setdiff1d(first_valid_eq_ids, drop_ids,
                                    assume_unique=True)
  
  # Same logic as in test to generate the gap predicted probabilities
  x_valid = GAP_DATA.iloc[first_valid_eq_ids]
#  x_valid = x_valid[:600000]
  num_valid_files = int(x_valid.shape[0]/valid_file_steps)
  x_valid = x_valid.iloc[np.arange(valid_file_steps*num_valid_files)]
  valid_ranges = (valid_file_steps*np.arange(num_valid_files),
                  valid_file_steps*(1+np.arange(num_valid_files))-(
                      hyperpars['block_steps']))
  (x_valid_batched, _, valid_start_rows) = utils.get_gap_prediction_features(
      x_valid, valid_ranges, hyperpars, order_start_rows=True)
  file_names = ['valid_' + str(i+1) for i in range(num_valid_files)]
  valid_preds = make_predictions(model_path, split, fold=0, num_folds=1,
                                 x_features=x_valid_batched)
  valid_gap_preds_aligned = utils.align_test_gap_preds(
      valid_preds, valid_file_steps, valid_start_rows, hyperpars, file_names)
  data_path = save_path + '_aligned_predictions_valid' + '.csv'
  valid_gap_preds_aligned.to_csv(data_path, index=False)
  
  
# Generate and store gap validation predictions for the train data
def train_save_gap_preds(model_path, save_path, split, hyperpars):
  # Same logic as in test to generate the gap predicted probabilities
  x_train = GAP_DATA
#  x_train = x_train[:600000]
  train_ranges = (np.array([0]),
                  np.array([x_train.shape[0] - hyperpars['block_steps']]))
  (x_train_batched, _, train_start_rows) = utils.get_gap_prediction_features(
      x_train, train_ranges, hyperpars, order_start_rows=True)
  train_preds = make_predictions(model_path, split, fold=0, num_folds=1,
                                 x_features=x_train_batched)
  train_gap_preds_aligned = utils.align_test_gap_preds(
      train_preds, x_train.shape[0], train_start_rows, hyperpars)
  data_path = save_path + '_aligned_predictions_train' + '.csv'
  train_gap_preds_aligned.to_csv(data_path, index=False)
    

##########################################
# Main logic for training and validation #
##########################################
mode = ['train', 'validation', 'train_validate', 'test', 'validate_save',
        'train_save', 'no_action'][4]
splits = [0]
model_description = 'initial_gap'
custom_model = models.initial_gap
overwrite_train = True
model_on_all_data = False
max_fold = 1
use_best_model_validation_test = True

hyperpars = {
    'block_steps': 2000,
    'epochs': 100,
    'steps_per_epoch': 2000,
    'max_gap_shifts': 1, # For oversampling of gaps in the training data
    'initial_lr': 1e-3,
    'batch_size': 32,
    'es_patience': 20,
    'reduce_lr_patience': 3,
    'drop_extreme_part_loss_frac': 0.1,
    'recurrent_cells': [32, 32, 32],
    'prediction_layers': [64, 32, 10],
    'gru_dropout': 0.25,
    'prediction_dropout': 0.25,
}

if use_best_model_validation_test and mode in [
    'validation', 'test', 'validate_save', 'train_save']:
  model_description = 'Best models/gap_model'
model_path = '/home/tom/Kaggle/LANL/Models/' + model_description
if mode in ['test', 'validate_save', 'train_save']:
  save_path = data_folder + "gap_model"
else:
  save_path = model_path

if mode == 'train':
  train_models(custom_model, save_path, splits, hyperpars, overwrite_train,
               model_on_all_data, max_fold)
elif mode == 'validation':
  valid_maes = validate_models(save_path, splits, max_fold)
elif mode == 'train_validate':
  train_models(custom_model, save_path, splits, hyperpars, overwrite_train,
               model_on_all_data, max_fold)
  print("\n")
  validate_models(save_path, splits, max_fold)
elif mode == 'test':
  test_model(model_path, save_path, splits[0], hyperpars)
elif mode == 'validate_save':
  validate_save_gap_preds(model_path, save_path, splits[0], hyperpars)
elif mode == 'train_save':
  train_save_gap_preds(model_path, save_path, splits[0], hyperpars)
