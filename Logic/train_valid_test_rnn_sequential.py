import warnings
with warnings.catch_warnings():
  warnings.simplefilter('ignore', RuntimeWarning)
  import models
import datetime
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import preprocess
from scipy.signal import savgol_filter
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
FIGURES_FOLDER = data_folder + 'rnn_sequential_pred_figures'

# Set global seed in order to obtain reproducible results
GLOBAL_SEED = 14
utils.make_results_reproducible(K, GLOBAL_SEED)

# Load the global variables TRAIN_DATA and TEST_DATA if they have not
# been loaded before.
if ((not 'TRAIN_DATA' in locals()) and (
     not 'TRAIN_DATA' in globals())) or (
    (not 'TEST_DATA' in locals()) and (
     not 'TEST_DATA' in globals())):
  TRAIN_DATA = pd.read_csv(
      data_folder + 'train_main_cpc_encodings_and_features.csv')
  TEST_DATA = pd.read_csv(
      data_folder + 'test_main_cpc_encodings_and_features.csv')
  TRAIN_DATA.sub_frac_greater_1000 = 0
  TEST_DATA.sub_frac_greater_1000 = 0
  
#  TRAIN_DATA_WIDE = pd.read_csv(
#      data_folder + 'train_features_scaled_keep_incomplete_target_quantile.csv')
#  TEST_DATA_WIDE = pd.read_csv(
#      data_folder + 'test_features_scaled_keep_incomplete_target_quantile.csv')
#  TRAIN_DATA_WIDE_UNSCALED = pd.read_csv(
#      data_folder + 'train_features_keep_incomplete_target_quantile.csv')
#  TEST_DATA_WIDE_UNSCALED = pd.read_csv(
#      data_folder + 'test_features_keep_incomplete_target_quantile.csv')
 
# Compute the global variable TRAIN_EQ_IDS if it has not been loaded before
if not 'TRAIN_EQ_IDS' in locals() and not 'TRAIN_EQ_IDS' in globals():
  TRAIN_EQ_IDS = 1+np.where(
      np.diff(TRAIN_DATA.notrain_target_original.values) > 0)[0]
  TRAIN_EQ_IDS = np.insert(TRAIN_EQ_IDS, 0, 0)
  TRAIN_EQ_IDS = np.append(TRAIN_EQ_IDS, TRAIN_DATA.shape[0])


# Helper function for naming the model path given the fold id
def get_fold_description(fold, num_folds):
  return fold if fold < num_folds else 'all_train'

# Fit a selected model and fold
def fit_model(custom_model, save_path_orig, fold, num_folds,
              skip_last_train_fold, train_val_split, hyperpars,
              overwrite_train, target_quantile, train_last_six_complete):
  if skip_last_train_fold and fold == num_folds-1:
    print('Skipping the last train fold')
    return
  fold_description = get_fold_description(fold, num_folds)
  save_path = '{}-{}.h5'.format(save_path_orig, fold_description)
  if not os.path.exists(save_path) or overwrite_train:
    if fold == num_folds:
      eq_ids = np.arange(TRAIN_EQ_IDS.shape[0])
    else:
      if train_last_six_complete:
        eq_ids = np.arange(10, 17)
      else:
        eq_ids = np.arange(train_val_split[fold][2][0],
                           train_val_split[fold][2][1]+1)
    if REMOVE_INCOMPLETE_EQS:
      eq_ids = eq_ids[np.logical_and(eq_ids > 0, eq_ids < 16)]
    print('EQ ids: {}'.format(eq_ids))
    train_ranges = (TRAIN_EQ_IDS[eq_ids], TRAIN_EQ_IDS[eq_ids+1])

    (inputs, outputs) = custom_model(hyperpars)
    model = Model(inputs=[inputs], outputs=[outputs])
    train_gen = utils.generator_rnn_sequential_batch(
        TRAIN_DATA, train_ranges, hyperpars, target_quantile)
    adam = Adam(lr=hyperpars['initial_lr'])
    train_loss = utils.wrapped_partial(
        utils.mae_last_part_loss, drop_target_part=(
        1-hyperpars['train_last_part_loss_frac']))
    model.compile(optimizer=adam, loss=train_loss, metrics=[])
    validation_monitor = 'loss'
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
#    callbacks.append(utils.PlotLosses())
#    callbacks.append(TensorBoard('./Graph'))
    
    model.fit_generator(train_gen,
                        steps_per_epoch=hyperpars['steps_per_epoch'],
                        epochs=hyperpars['epochs'],
                        callbacks=callbacks,
                        validation_data=None,
                        )
    
  # Evaluate OOF performance
  fold_mae = get_fold_mae(save_path_orig, fold, num_folds, train_val_split,
                          hyperpars, target_quantile,
                          hyperpars['train_valid_batch'])
  
  return fold_mae
  

# Train all folds
def train_models(custom_model, save_path, hyperpars, overwrite_train,
                 train_on_all_data, remove_overlap_chunks,
                 train_all_previous, skip_last_train_fold, target_quantile,
                 train_last_six_complete=False):
  train_val_split = preprocess.train_val_split(
      remove_overlap_chunks, ordered=True, num_folds=NUM_FOLDS,
      remove_incomplete_eqs=REMOVE_INCOMPLETE_EQS,
      train_all_previous=train_all_previous,
      target_quantile=target_quantile)
  num_folds = len(train_val_split) if not train_last_six_complete else 1
  num_train_models = num_folds + int(train_on_all_data)
  for fold in range(num_train_models):
    print('\nProcessing fold {} of {}'.format(fold+1, num_train_models))
    K.clear_session()
    fit_model(custom_model, save_path, fold, num_folds, skip_last_train_fold,
              train_val_split, hyperpars, overwrite_train, target_quantile,
              train_last_six_complete)
      
# Obtain the OOF validation score for a given model
def get_fold_mae(save_path_orig, fold, num_folds, train_val_split, hyperpars,
                 target_quantile, validation_batch, plot_error=True):
  # Get the validation ranges
  if fold == num_folds:
    valid_eq_ids = np.arange(TRAIN_EQ_IDS.shape[0])
  else:
    if train_val_split[fold][3].size:
      valid_eq_ids = np.arange(train_val_split[fold][3][0],
                               train_val_split[fold][3][-1]+1)
    else:
      valid_eq_ids = np.array([])
#  import pdb; pdb.set_trace()
#  valid_eq_ids = np.arange(1, 3)
#  valid_eq_ids = np.arange(14, 17)
#  valid_eq_ids = np.arange(11, 17)
  if REMOVE_INCOMPLETE_EQS:
    valid_eq_ids = valid_eq_ids[np.logical_and(valid_eq_ids > 0,
                                               valid_eq_ids < 16)]
  
  if valid_eq_ids.size:
    valid_ranges = (TRAIN_EQ_IDS[valid_eq_ids], TRAIN_EQ_IDS[valid_eq_ids+1])
    
    valid_gen = utils.generator_rnn_sequential_batch(
        TRAIN_DATA, valid_ranges, hyperpars, target_quantile)
    
    # Generate the validation data by calling the generator *N* times
    valid_data = list(itertools.islice(valid_gen, validation_batch))
    x_valid, y_valid, valid_start_rows = zip(*valid_data)
    x_valid = np.concatenate(x_valid)
    y_valid_orig = np.concatenate(y_valid).mean(1)
    valid_start_rows = np.concatenate(valid_start_rows)
    x_valid_rep = np.repeat(x_valid, 1, 0)
#    K.clear_session() # DO NOT UNCOMMENT
    valid_preds_orig = make_predictions(
        save_path_orig, hyperpars, fold, num_folds, x_valid_rep)
    valid_preds_orig = np.median(
        valid_preds_orig.reshape(x_valid.shape[0], -1), 1)
    if target_quantile:
#      test_cols = [c for c in TEST_DATA.columns if not c=='target' and not 'notrain' in c]
#      if not hyperpars['include_freq_features']:
#        test_cols = [c for c in test_cols if c[:4] != 'sub_']
#      if not hyperpars['include_cpc_features']:
#        test_cols = [c for c in test_cols if c[:4] != 'enc_']
#      x_test = TEST_DATA[test_cols].values.reshape(2624, 100, -1)
##      x_test = TRAIN_DATA[test_cols].values.reshape(4184, 100, -1)
#      test_preds = make_predictions(
#        save_path_orig, hyperpars, fold, num_folds, x_test)
#      import pdb; pdb.set_trace()
      valid_preds = train_val_split[fold][7] * (1-valid_preds_orig)
#      valid_preds = train_val_split[4][7] * (1-valid_preds_orig)
#      valid_preds = train_val_split[3][7] * (1-valid_preds_orig)
      y_valid = TRAIN_DATA.notrain_target_original.values[valid_start_rows]
    else:
      valid_preds = valid_preds_orig
      y_valid = y_valid_orig
    oof_mae = np.abs(valid_preds - y_valid).mean()
    
    # Don't normalize the mean when using the quantile method
    if target_quantile:
      valid_preds_norm = valid_preds
    else:
#      mean_offset = train_val_split[fold][6] - valid_preds.mean()
      mean_offset = train_val_split[fold][7]/2 - valid_preds.mean()
      valid_multiplier = 1 + mean_offset/valid_preds.mean()
      valid_preds_norm = valid_preds*valid_multiplier
#    oof_mae_norm = np.abs(valid_preds+mean_offset- y_valid).mean()
    oof_mae_norm = np.abs(valid_preds_norm - y_valid).mean()
    valid_size = x_valid.size
  else:
    oof_mae = 0
    oof_mae_norm = 0
    valid_size = 0
  
  error_description = 'OOF MAE' if fold < num_folds else 'Train error'
  print('{}: {}'.format(error_description, np.round(oof_mae, 3)))
  print('{}: {}'.format(error_description+' NORM', np.round(oof_mae_norm, 3)))
  
  # Plot the out of fold predictions versus the ground truth
  if plot_error and valid_eq_ids.size:
    quant_ext = '_quantile' if target_quantile else '_ttf'
    filename = os.path.join(
        FIGURES_FOLDER, 'oof_preds_' + str(fold) + quant_ext + '.png')
    ordered_ids = valid_start_rows.argsort()
    valid_ids_ordered = valid_start_rows[ordered_ids]
    valid_preds_ordered = valid_preds[ordered_ids]
    valid_preds_ordered_sg = savgol_filter(valid_preds_ordered, 251, 3)
    valid_preds_norm_ordered = valid_preds_norm[ordered_ids]
    valid_preds_norm_ordered_sg = savgol_filter(
        valid_preds_norm_ordered, 251, 3)
    y_valid_ordered = y_valid[ordered_ids]
    plt.figure(figsize=(10, 8))
    plt.plot(valid_ids_ordered, valid_preds_ordered_sg)
    plt.plot(valid_ids_ordered, valid_preds_norm_ordered_sg,
             linestyle='dashed', color='blue')
    plt.plot(valid_ids_ordered, y_valid_ordered, color='green')
    plt.hlines(valid_preds.min(), valid_start_rows.min(),
               valid_start_rows.max(), color='red', linestyles='dashed')
    plt.hlines(valid_preds.max(), valid_start_rows.min(),
               valid_start_rows.max(), color='red', linestyles='dashed')
    plt.savefig(filename)
    plt.show()
    
  return oof_mae, oof_mae_norm, valid_size


def make_predictions(save_path_orig, hyperpars, fold, num_folds, x_features):
  fold_description = get_fold_description(fold, num_folds)
  save_path = '{}-{}.h5'.format(save_path_orig, fold_description)
  with warnings.catch_warnings():
    warnings.simplefilter('ignore', RuntimeWarning)
    model = load_model(save_path, custom_objects={
            'mae_last_part_loss': utils.mae_last_part_loss})
  preds = model.predict(x_features, verbose=1)
  num_steps = preds.shape[1]
  pred_start_step = int((1-hyperpars['predict_last_part_frac']) * num_steps)
  aggregate_preds = preds[:, pred_start_step:].mean(1)
#  import pdb; pdb.set_trace()
  return aggregate_preds


# Validate all folds
def validate_models(save_path, hyperpars, remove_overlap_chunks,
                    train_all_previous, target_quantile):
  train_val_split = preprocess.train_val_split(
      remove_overlap_chunks, ordered=True, num_folds=NUM_FOLDS,
      remove_incomplete_eqs=REMOVE_INCOMPLETE_EQS,
      train_all_previous=train_all_previous,
      target_quantile=target_quantile)
  
  num_folds = len(train_val_split)
  sum_maes = []
  maes = []
  fold_mae_norms = []
  total_count = 0
  for fold in range(num_folds):
    print('Processing fold {} of {}'.format(fold+1, num_folds))
    fold_mae, fold_mae_norm, oof_count = get_fold_mae(
        save_path, fold, num_folds, train_val_split, hyperpars,
        target_quantile, hyperpars['validation_valid_batch'])
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
    
# Test prediction generation
def test_model(save_path, test_on_all_folds, train_all_previous,
               target_quantile, median_test_cyle_length, seed_ext=None,
               train_last_six_complete=False, drop_first_test_fold=False):
  train_val_split = preprocess.train_val_split(
      ordered=True, remove_overlap_chunks=True, num_folds=NUM_FOLDS,
      remove_incomplete_eqs=REMOVE_INCOMPLETE_EQS,
      train_all_previous=train_all_previous,
      target_quantile=target_quantile)
  
  test_file_steps = int(150000/hyperpars['block_steps'])
  num_test_files = int(TEST_DATA.shape[0]/test_file_steps)
  test_ranges = (test_file_steps*np.arange(num_test_files),
                 test_file_steps*(1+np.arange(num_test_files))-(
                     hyperpars['chunk_blocks']))
  (x_test_batched, test_start_rows) = utils.get_rnn_prediction_features(
      TEST_DATA, test_ranges, hyperpars, order_start_rows=True)
  x_test_batched = np.repeat(x_test_batched, 20, 0)
  
  num_folds = len(train_val_split)
  num_folds = 1 if train_last_six_complete else num_folds
  pred_folds = [f for f in range(num_folds)] if test_on_all_folds else [
      num_folds-1]
  model_preds = np.zeros((num_test_files, len(pred_folds)))
  for (i, fold) in enumerate(pred_folds):
    print('Making test predictions {} of {}'.format(i+1,
          len(pred_folds)))
#    K.clear_session() # DO NOT UNCOMMENT
    fold_test_preds = make_predictions(save_path, hyperpars, fold, num_folds,
                                       x_test_batched)
    model_preds[:, fold] = np.mean(
        fold_test_preds.reshape(num_test_files, -1), 1)
    
  model_preds = model_preds[:, 1:] if drop_first_test_fold else model_preds
  preds_test = np.median(model_preds, 1)
  
  if target_quantile:
    preds_test = median_test_cyle_length*(1-preds_test)
  
  # Write the output pandas data frame
  submission = pd.read_csv(data_folder + 'sample_submission.csv')
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
train_all_previous = True
skip_last_train_fold = False # Keep False when preparing a test submission!!!
median_test_cyle_length = [8, 9.7, 11.0, 12.0][3] # public LB and private LB values
target_quantile = True

model_description = 'sequential_gru'
custom_model = models.sequential_gru
overwrite_train = True
train_on_all_data = False
REMOVE_INCOMPLETE_EQS = False # Delete train_val_split.pickle after changing
test_on_all_folds = True
remove_overlap_chunks = True
train_test_num_eqs = [3, 4][0]
NUM_FOLDS = 6 if train_test_num_eqs == 3 else 5
num_final_test_seeds = 10

hyperpars = {
    'block_steps': 1500,
    'chunk_blocks': int(150000*1.0/1500),
    'input_dimension_freq': 40,
    'input_dimension_cpc': 16,
    'include_cpc_features': False,
    'include_freq_features': True,
    'epochs': 10,
    'steps_per_epoch': 1000,
#    'validation_split': 0.2,
    
    'initial_lr': 1e-4,
    'es_patience': 20,
    'batch_size': 32,
    'reduce_lr_patience': 2,
    'train_last_part_loss_frac': 0.4,
    'predict_last_part_frac': 0.1,
    'encoding_layers': [32],
    'relu_last_encoding_layer': True,
    'encoding_input_dropout': 0,
    'encoding_dropout': 0.25,
    'recurrent_cells': [32],
    'gru_dropout': 0,
    'prediction_layers': [],
    'clip_preds_zero_one': target_quantile,
    
    'train_valid_batch': 1000,
    'validation_valid_batch': 2000,
}

save_path = '/home/tom/Kaggle/LANL/Models/' + model_description
if mode == 'train':
  train_models(custom_model, save_path, hyperpars, overwrite_train,
               train_on_all_data, remove_overlap_chunks, train_all_previous,
               skip_last_train_fold, target_quantile)
elif mode == 'validation':
  valid_maes = validate_models(save_path, hyperpars, remove_overlap_chunks,
                               train_all_previous, target_quantile)
elif mode == 'train_validate':
  train_models(custom_model, save_path, hyperpars, overwrite_train,
               train_on_all_data, remove_overlap_chunks, train_all_previous,
               skip_last_train_fold, target_quantile)
  print('\n')
  validate_models(save_path, hyperpars, remove_overlap_chunks,
                  train_all_previous, target_quantile)
elif mode == 'test':
  test_model(save_path, test_on_all_folds, train_all_previous,
             target_quantile, median_test_cyle_length)
  
elif mode == 'final_test':
  # Models trained on the last 6 complete cycles
  for i in range(num_final_test_seeds):
    print('\nFinal test predictions {} of {}'.format(
        i+1, num_final_test_seeds))
    hyperpars['random_seed'] = i+1
    NUM_FOLDS = 1
    train_models(custom_model, save_path, hyperpars, overwrite_train=True,
                 train_on_all_data=False, remove_overlap_chunks=False,
                 train_all_previous=False, skip_last_train_fold=False,
                 target_quantile=True, train_last_six_complete=True)
    
    test_model(save_path, test_on_all_folds=False, train_all_previous=False,
               target_quantile=True,
               median_test_cyle_length=median_test_cyle_length,
               seed_ext=' - nn_last_six_seed_' + str(i+1) + '_of_' + str(
                   num_final_test_seeds) + '_' + str(median_test_cyle_length),
                   train_last_six_complete=True)
               
  # Models trained on all but the first folds using all data up to that point
  for i in range(num_final_test_seeds):
    print('\nFinal test predictions {} of {}'.format(
        i+1, num_final_test_seeds))
    hyperpars['random_seed'] = i+1
    NUM_FOLDS = 6 if train_test_num_eqs == 3 else 5
    train_models(custom_model, save_path, hyperpars, overwrite_train=True,
                 train_on_all_data=False, remove_overlap_chunks=False,
                 train_all_previous=True, skip_last_train_fold=False,
                 target_quantile=True, train_last_six_complete=False)
    
    test_model(save_path, test_on_all_folds=True, train_all_previous=True,
               target_quantile=True,
               median_test_cyle_length=median_test_cyle_length,
               seed_ext=' - nn_all_prev_seed_' + str(i+1) + '_of_' + str(
                   num_final_test_seeds) + '_' + str(median_test_cyle_length),
                   train_last_six_complete=False, drop_first_test_fold=True)
