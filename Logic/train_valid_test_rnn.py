import warnings
with warnings.catch_warnings():
  warnings.simplefilter('ignore', RuntimeWarning)
  import models
import datetime
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

# Set global seed in order to obtain reproducible results
GLOBAL_SEED = 14
utils.make_results_reproducible(K, GLOBAL_SEED)


# Helper function for naming the model path given the fold id
def get_fold_description(fold, num_folds):
  return fold if fold < num_folds else 'all_train'

# Fit a selected model and fold
def fit_model(custom_model, save_path_orig, split, fold, num_folds,
              train_val_split, hyperpars, overwrite_train, features,
              other_features, targets):
  fold_description = get_fold_description(fold, num_folds)
  save_path = '{}-{}-{}.h5'.format(save_path_orig, split, fold_description)
  if not os.path.exists(save_path) or overwrite_train:
    if fold == num_folds:
      train_ids = np.random.permutation(np.arange(features.shape[0]))
    else:
      train_ids = train_val_split[split][fold][0]
    (x_train, y_train) = utils.reshape_time_dim(features, targets, train_ids)
    
    (inputs, outputs) = custom_model(
        x_train.shape[1], x_train.shape[2], hyperpars)
    model = Model(inputs=[inputs], outputs=[outputs])
    adam = Adam(lr=hyperpars['initial_lr'])
    train_loss = utils.wrapped_partial(
        utils.mae_last_part_loss, drop_target_part=(
        1-hyperpars['train_last_part_loss_frac']))
    model.compile(optimizer=adam, loss=train_loss, metrics=[])
    validation_monitor = 'val_loss'
    earlystopper = EarlyStopping(monitor=validation_monitor, mode='min',
                                 patience=hyperpars['es_patience'], verbose=1)
    checkpointer = ModelCheckpoint(save_path, monitor=validation_monitor,
                                   mode='min', verbose=1, save_best_only=True,
                                   period=1)
    reduce_lr = ReduceLROnPlateau(factor=1/math.sqrt(10), verbose=1,
                                  patience=hyperpars['reduce_lr_patience'],
                                  min_lr=hyperpars['initial_lr']/100,
                                  monitor=validation_monitor, mode='min')
    callbacks = [earlystopper, checkpointer, reduce_lr]
    callbacks.append(utils.PlotLosses())
#    callbacks.append(TensorBoard('./Graph'))
    
    model.fit(x_train, y_train, validation_split=hyperpars['validation_split'],
              batch_size=hyperpars['batch_size'], epochs=hyperpars['epochs'],
              sample_weight=other_features.train_weight.values[train_ids],
              callbacks=callbacks)
    
  # Evaluate OOF performance
  fold_mae = get_fold_mae(save_path_orig, split, fold, num_folds,
                          train_val_split, features, targets, hyperpars)
  
  return fold_mae
  

# Train all folds and random splits
def train_models(custom_model, save_path, splits, hyperpars, overwrite_train,
                 model_on_all_data, remove_overlap_chunks):
  (features, other_features, targets) = preprocess.get_preprocessed(
      'train', remove_overlap_chunks, scale=True)
  
  train_val_split = preprocess.train_val_split(remove_overlap_chunks)
  num_folds = len(train_val_split[0])
  num_train_models = num_folds + int(model_on_all_data)
  for split_id, split in enumerate(splits):
    print('Processing split {} of {}'.format(split_id+1, len(splits)))
    for fold in range(num_train_models):
      print('\nProcessing fold {} of {}'.format(fold+1, num_train_models))
      K.clear_session()
      fit_model(custom_model, save_path, split, fold, num_folds,
                train_val_split, hyperpars, overwrite_train, features,
                other_features, targets)
      
# Obtain the OOF validation score for a given model
def get_fold_mae(save_path_orig, split, fold, num_folds, train_val_split,
                 features, targets, hyperpars):
  valid_ids = train_val_split[split][fold][1] if fold < num_folds else (
      np.arange(features.shape[0]))
  x_valid, y_valid = utils.reshape_time_dim(features, targets, valid_ids)
  valid_preds = make_predictions(save_path_orig, split, fold, num_folds,
                                 x_valid)
  oof_mae = np.abs(valid_preds - y_valid[:, 0]).mean()
  error_description = "OOF MAE" if fold < num_folds else "Train error"
  print('{}: {}'.format(error_description, np.round(oof_mae, 3)))
#  import pdb; pdb.set_trace()
  
  return oof_mae, valid_ids.size


def make_predictions(save_path_orig, split, fold, num_folds, x_features):
  fold_description = get_fold_description(fold, num_folds)
  save_path = '{}-{}-{}.h5'.format(save_path_orig, split, fold_description)
  with warnings.catch_warnings():
    warnings.simplefilter('ignore', RuntimeWarning)
    model = load_model(save_path, custom_objects={
            'mae_last_part_loss': utils.mae_last_part_loss})
  preds = model.predict(x_features, verbose=1)
  num_steps = preds.shape[1]
  pred_start_step = int((1-hyperpars['predict_last_part_frac']) * num_steps)
  aggregate_preds = preds[:, pred_start_step:].mean(1)
  return aggregate_preds

# Validate all folds and random splits
def validate_models(save_path, splits, remove_overlap_chunks):
  (features, other_features, targets) = preprocess.get_preprocessed(
      'train', remove_overlap_chunks, scale=True)
  
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
          targets, hyperpars)
      sum_split_maes.append(fold_mae*oof_count)
      split_maes.append((fold_mae, oof_count))
    split_mae = np.array(sum_split_maes).sum()/features.shape[0]
    split_maes = [split_mae] + split_maes
    print('Split OOF MAE: {}'.format(np.round(split_mae, 3)))
    valid_maes.append((split_maes, splits[split_id]))
  return valid_maes
    
# Test prediction generation - only consider first split for now
def test_model(save_path, split, model_on_all_data):
  (x_test, other_test, _) = preprocess.get_preprocessed(
      'test', remove_overlap_chunks=True, scale=True)
  (x_test_reshaped, _) = utils.reshape_time_dim(
      x_test, np.zeros_like(x_test), np.arange(x_test.shape[0]))
  train_val_split = preprocess.train_val_split(remove_overlap_chunks=True)
  num_folds = len(train_val_split[split])
  num_test = x_test.shape[0]
  num_prediction_models = num_folds + int(model_on_all_data)
  model_preds = np.zeros((num_test, num_prediction_models))
  for fold in range(num_prediction_models):
    print("Making test predictions {} of {}".format(fold+1,
          num_prediction_models))
    model_preds[:, fold] = make_predictions(save_path, split, fold, num_folds,
               x_test_reshaped)
  
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
splits = [0]
model_description = 'initial_gru'
custom_model = models.initial_gru
overwrite_train = True
model_on_all_data = True
remove_overlap_chunks = True

hyperpars = {
    'epochs': 50,
    'validation_split': 0.2,
    'es_patience': 20,
    'initial_lr': 1e-3,
    'batch_size': 32,
    'reduce_lr_patience': 5,
    'train_last_part_loss_frac': 0.4,
    'predict_last_part_frac': 0.1,
    'encoding_layers': [32, 32],
    'encoding_input_dropout': 0.2,
    'num_recurrent_cells': 256,
    'prediction_layers': [10],
}

save_path = '/home/tom/Kaggle/LANL/Models/' + model_description
if mode == 'train':
  train_models(custom_model, save_path, splits, hyperpars, overwrite_train,
               model_on_all_data, remove_overlap_chunks)
elif mode == 'validation':
  valid_maes = validate_models(save_path, splits, remove_overlap_chunks)
elif mode == 'train_validate':
  train_models(custom_model, save_path, splits, hyperpars, overwrite_train,
               model_on_all_data, remove_overlap_chunks)
  print("\n")
  validate_models(save_path, splits, remove_overlap_chunks)
elif mode == 'test':
  test_model(save_path, splits[0], model_on_all_data)
