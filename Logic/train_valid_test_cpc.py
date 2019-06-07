import warnings
with warnings.catch_warnings():
  warnings.simplefilter('ignore', RuntimeWarning)
  import models
import gc
import itertools
import math
import numpy as np
import os
import pandas as pd
import preprocess
import tensorflow as tf
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

# Load the global variables TRAIN_AUGMENT and TEST_AUGMENT if they have not
# been loaded before.
if ((not 'TRAIN_AUGMENT' in locals()) and (
     not 'TRAIN_AUGMENT' in globals())) or (
    (not 'TEST_AUGMENT' in locals()) and (
     not 'TEST_AUGMENT' in globals())):
  TRAIN_AUGMENT = pd.read_csv(data_folder + 'train_normalized_gap_augment.csv',
                              dtype=np.float16)
  TEST_AUGMENT = pd.read_csv(data_folder + 'test_normalized_gap_augment.csv',
                             dtype=np.float16)
 
# Compute the global variable TRAIN_EQ_IDS if it has not been loaded before
if not 'TRAIN_EQ_IDS' in locals() and not 'TRAIN_EQ_IDS' in globals():
  TRAIN_EQ_IDS = 1+np.where(
      np.diff(TRAIN_AUGMENT.time_to_failure.values) > 0)[0]
  TRAIN_EQ_IDS = np.insert(TRAIN_EQ_IDS, 0, 0)
  TRAIN_EQ_IDS = np.append(TRAIN_EQ_IDS, TRAIN_AUGMENT.shape[0])

# Helper function for naming the model path given the fold id
def get_fold_description(fold, num_folds):
  return fold if fold < num_folds else 'all_train'

# Fit a selected model and fold
def fit_model(custom_model, save_path_orig, split, fold, num_folds,
              train_val_split, hyperpars, overwrite_train,
              remove_incomplete_eqs):
  fold_description = get_fold_description(fold, num_folds)
  save_path = '{}-{}-{}.h5'.format(save_path_orig, split, fold_description)
  save_path_encoder = save_path[:-3] + '_encoder.h5'
  if not os.path.exists(save_path) or overwrite_train:
    if fold == num_folds:
      if remove_incomplete_eqs:
        train_ranges = (np.array([TRAIN_EQ_IDS[0]]),
                        np.array([TRAIN_EQ_IDS[-1]]))
      else:
        train_ranges = (np.array([0]), np.array([TRAIN_AUGMENT.shape[0]]))
    else:
      train_ranges = train_val_split[split][fold][0]
    train_gen = utils.generator_cpc_batch(TRAIN_AUGMENT, TEST_AUGMENT,
                                          train_ranges, TRAIN_EQ_IDS,
                                          hyperpars)
    
    (inputs, outputs, encoder) = custom_model(hyperpars)
    model = Model(inputs=inputs, outputs=outputs)
    adam = Adam(lr=hyperpars['initial_lr'])
    model.compile(optimizer=adam, loss=LOSS_TYPES,
                  loss_weights=hyperpars['loss_weights'], metrics=[])
    (monitor, monitor_mode) = ('loss', 'min')
    earlystopper = EarlyStopping(
        monitor=monitor, mode=monitor_mode,
        patience=hyperpars['es_patience'], verbose=1)
    checkpointer = ModelCheckpoint(save_path, monitor=monitor,
                                   mode=monitor_mode, verbose=1,
                                   save_best_only=True, period=1)
    encoder_checkpointer = utils.EncoderCheckpointer(
        save_path_encoder, encoder, monitor=monitor, mode=monitor_mode,
        save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(factor=1/math.sqrt(10), verbose=1,
                                  patience=hyperpars['reduce_lr_patience'],
                                  min_lr=hyperpars['initial_lr']/100,
                                  monitor=monitor,
                                  mode=monitor_mode)
    callbacks = [earlystopper, checkpointer, encoder_checkpointer, reduce_lr]
    callbacks.append(utils.PlotLosses())
#    callbacks.append(TensorBoard('./Graph'))
    
    model.fit_generator(train_gen,
                        steps_per_epoch=hyperpars['steps_per_epoch'],
                        epochs=hyperpars['epochs'],
                        callbacks=callbacks,
                        validation_data=None,
                        )
    
  # Evaluate the trained model performance
  validate_model(save_path_orig, split, fold, num_folds, train_val_split,
                 hyperpars, hyperpars['train_valid_batch'],
                 remove_incomplete_eqs)

# Train all folds and random splits
def train_models(custom_model, save_path, splits, hyperpars, overwrite_train,
                 model_on_all_data, model_on_all_data_only, max_fold,
                 remove_incomplete_eqs):
  train_val_split = preprocess.train_val_split_gaps()
  num_folds = len(train_val_split[0])
  num_train_models = num_folds + int(model_on_all_data)
  if not model_on_all_data_only:
    num_train_models = min(max_fold, num_train_models)
  for split_id, split in enumerate(splits):
    print('Processing split {} of {}'.format(split_id+1, len(splits)))
    for fold in range(num_train_models):
      if not model_on_all_data_only or fold == num_folds:
        print('\nProcessing fold {} of {}'.format(fold+1, num_train_models))
        K.clear_session()
        fit_model(custom_model, save_path, split, fold, num_folds,
                  train_val_split, hyperpars, overwrite_train,
                  remove_incomplete_eqs)
      
# Obtain the OOF validation performance of the different modeling targets
def validate_model(save_path_orig, split, fold, num_folds, train_val_split,
                   hyperpars, validation_batch, remove_incomplete_eqs):
  if fold < num_folds:
    valid_ranges = train_val_split[split][fold][1]
  else:
    if remove_incomplete_eqs:
      valid_ranges = (np.array([TRAIN_EQ_IDS[0]]),
                      np.array([TRAIN_EQ_IDS[-1]]))
    else:
      valid_ranges = (np.array([0]), np.array([TRAIN_AUGMENT.shape[0]]))
  valid_gen = utils.generator_cpc_batch(TRAIN_AUGMENT, TEST_AUGMENT,
                                        valid_ranges, TRAIN_EQ_IDS, hyperpars)
  
  # Generate the validation data by calling the generator *N* times
  valid_data = list(itertools.islice(valid_gen, validation_batch))
  valid_preds = make_predictions(save_path_orig, split, fold, num_folds,
                                 valid_data)
  target_ttf = np.concatenate([d[1][0] for d in valid_data])
  valid_mae = np.abs(valid_preds[0]-target_ttf).mean()
  error_description = "OOF MAE" if fold < num_folds else "Train MAE"
#  import pdb; pdb.set_trace()
  print('{0}: {1:.3f}'.format(error_description, valid_mae))
  
  return valid_mae

# Helper function for generating model predictions
def make_predictions(save_path_orig, split, fold, num_folds, data, model=None):
  if model is None:
    fold_description = get_fold_description(fold, num_folds)
    save_path = '{}-{}-{}.h5'.format(save_path_orig, split, fold_description)
    with warnings.catch_warnings():
      warnings.simplefilter('ignore', RuntimeWarning)
      model = load_model(save_path, custom_objects={
              'Attention': models.Attention,
              'GradientReversal': models.GradientReversal,
              'tf': tf,
              })
  #    encoder_model = load_model(save_path[:-3] + '_encoder.h5', custom_objects={
  #            'Attention': models.Attention,
  #            })
  model_inputs = []
  for i in range(len(data[0][0])):
    model_inputs.append(np.concatenate([d[0][i] for d in data]))
  preds = model.predict(model_inputs, verbose=1)
  
  return preds

# Validate all folds and random splits
def validate_models(save_path, splits, model_on_all_data_only, max_fold,
                    remove_incomplete_eqs):
  train_val_split = preprocess.train_val_split_gaps()
  num_folds = len(train_val_split[0])
  if model_on_all_data_only:
    num_validation_models = num_folds+1
  else:
    num_validation_models = min(max_fold, num_folds)
  valid_maes = []
  for split_id, split in enumerate(splits):
    print('Processing split {} of {}'.format(split_id+1, len(splits)))
    split_maes = []
    for fold in range(num_validation_models):
      if not model_on_all_data_only or fold == num_folds:
        print('Processing fold {} of {}'.format(fold+1, num_validation_models))
        fold_mae = validate_model(
            save_path, split, fold, num_folds, train_val_split, hyperpars,
            hyperpars['validation_valid_batch'], remove_incomplete_eqs)
        split_maes.append((fold_mae))
    split_mae = np.array(split_maes).mean()
    split_maes = [split_mae] + split_maes 
    print('Split OOF MAE: {0:.3f}'.format(split_mae))
    valid_maes.append((split_maes, splits[split_id]))
    
  return valid_maes

# Generate and store chunk order validation predictions for the first 11
# validation eqs.
# The resulting predictions are of approximately the same dimensions as the
# test predictions.
def valid_order(model_path, split, data_folder, hyperpars):
  # Determine the first eleven validation earthquake ids
  num_first_eqs = 11
  valid_file_steps = 150000
  comp_rows_per_it = 4
  train_val_split = preprocess.train_val_split_gaps()
  num_folds = len(train_val_split[0])
  
  
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
  VALID_DATA = TRAIN_AUGMENT.iloc[first_valid_eq_ids]
#  VALID_DATA = VALID_DATA[:(150000*16)]
  num_valid_files = int(VALID_DATA.shape[0]/valid_file_steps)
  VALID_DATA = VALID_DATA.iloc[np.arange(valid_file_steps*num_valid_files)]
  
  fold_description = get_fold_description(num_folds, num_folds)
  ENCODER_PATH = '{}-{}-{}.h5'.format(model_path, split, fold_description)
  encoder_model = load_model(ENCODER_PATH, custom_objects={
        'Attention': models.Attention,
        'GradientReversal': models.GradientReversal,
        'tf': tf,})
  num_iterations = int(num_valid_files/comp_rows_per_it)
  order_probs = np.zeros((num_valid_files, num_valid_files))
  
  for i in range(num_iterations):
    gc.collect()
    print('\nIteration {} of {}'.format(i+1, num_iterations))
    first_test_id = int(comp_rows_per_it*i)
    test_gen = utils.generator_cpc_batch_test(
        VALID_DATA, hyperpars, encoder_model, first_test_id=first_test_id)
  
    # Generate the test data by calling the generator *N* times
    N = int(num_valid_files/4*comp_rows_per_it)
    test_data = list(itertools.islice(test_gen, N))
    test_preds = make_predictions(model_path, split=-1, fold=-1, num_folds=-1,
                                  data=test_data, model=encoder_model)
    order_preds = test_preds[3][:, :, :4].mean(-1).reshape(
        [test_preds[3].shape[0], -1, 4]).mean(-1)
    order_probs[first_test_id:(first_test_id+comp_rows_per_it)] = (
        order_preds.reshape([comp_rows_per_it, -1]))
    
  save_path = data_folder + 'valid_order_probs_raw_signal.npy'
  np.save(save_path, order_probs) # np.load(save_path)
  

# Generate and store chunk order test predictions.
def test_order(model_path, split, data_folder, hyperpars):
  # Determine the first eleven validation earthquake ids
  comp_rows_per_it = 4
  train_val_split = preprocess.train_val_split_gaps()
  num_folds = len(train_val_split[0])
  num_test_files = 2624
  
  fold_description = get_fold_description(num_folds, num_folds)
  ENCODER_PATH = '{}-{}-{}.h5'.format(model_path, split, fold_description)
  encoder_model = load_model(ENCODER_PATH, custom_objects={
        'Attention': models.Attention,
        'GradientReversal': models.GradientReversal,
        'tf': tf,})
  num_iterations = int(num_test_files/comp_rows_per_it)
  order_probs = np.zeros((num_test_files, num_test_files))
  
  for i in range(num_iterations):
    gc.collect()
    print('\nIteration {} of {}'.format(i+1, num_iterations))
    first_test_id = int(comp_rows_per_it*i)
    test_gen = utils.generator_cpc_batch_test(
        TEST_AUGMENT, hyperpars, encoder_model, first_test_id=first_test_id)
  
    # Generate the test data by calling the generator *N* times
    N = int(num_test_files/4*comp_rows_per_it)
    test_data = list(itertools.islice(test_gen, N))
    test_preds = make_predictions(model_path, split=-1, fold=-1, num_folds=-1,
                                  data=test_data, model=encoder_model)
    order_preds = test_preds[3][:, :, :4].mean(-1).reshape(
        [test_preds[3].shape[0], -1, 4]).mean(-1)
    order_probs[first_test_id:(first_test_id+comp_rows_per_it)] = (
        order_preds.reshape([comp_rows_per_it, -1]))
    
  save_path = data_folder + 'test_order_probs_raw_signal.npy'
  np.save(save_path, order_probs) # np.load(save_path)


##########################################
# Main logic for training and validation #
##########################################
mode = ['train', 'validation', 'train_validate', 'valid_order', 'test_order',
        'no_action'][1]
splits = [0]
model_description = 'initial_cpc'
custom_model = models.initial_cpc
overwrite_train = True
model_on_all_data = True
model_on_all_data_only = True
remove_incomplete_eqs = True
max_fold = 1

# List the loss types for the different outputs
LOSS_TYPES = {
    'train_mae_prediction': 'mean_absolute_error',
    'domain_prediction': 'binary_crossentropy',
    'additional_domain_prediction': 'binary_crossentropy',
    'train_subchunk_predictions': 'binary_crossentropy',
    'train_chunk_predictions': 'categorical_crossentropy',
    'train_eq_predictions': 'categorical_crossentropy',
    'test_subchunk_predictions': 'binary_crossentropy',
    'test_chunk_predictions': 'categorical_crossentropy',
    }

hyperpars = {
    'block_steps': 1500,
    'epochs': 100,
    'steps_per_epoch': 1000,
    'batch_size': 32,
    'num_decoys': 4,
    
    'initial_lr': 1e-3,
    'es_patience': 20,
    'reduce_lr_patience': 4,
    'loss_weights': {
        'train_mae_prediction': 1.0,
        'domain_prediction': 0.3,
        'additional_domain_prediction': 0.3,
        'train_subchunk_predictions': 0.5,
        'train_chunk_predictions': 0.3,
        'train_eq_predictions': 0.3,
        'test_subchunk_predictions': 0.5,
        'test_chunk_predictions': 0.3,
        },
    
    'drop_gap_prediction_from_inputs': False,
    'use_attention_encoder': False,
    'grad_rev_lambda': 0.5, # -1 corresponds to no gradient reversal
    'encoder_embedding_layers': [16],
    'encoder_recurrent_cells': [32, 32],
    'mae_prediction_layers': [32, 16],
    'domain_prediction_layers': [32, 16],
    'additional_domain_prediction_layers': [32, 16],
    'subchunk_prediction_layers': [32, 32, 16],
    'chunk_prediction_layers': [16, 16],
    'eq_prediction_layers': [16, 16],
    'encoder_mlp_dropout': 0.25,
    'gru_dropout': 0.25,
    'prediction_dropout': 0.25,
    
    'train_valid_batch': 100,
    'validation_valid_batch': 2000,
    
    'test_predictions_per_chunk': 10,
    'test_offset_multiplier': int(4096/3/2),
}

model_path = '/home/tom/Kaggle/LANL/Models/' + model_description

if mode == 'train':
  train_models(custom_model, model_path, splits, hyperpars, overwrite_train,
               model_on_all_data, model_on_all_data_only, max_fold,
               remove_incomplete_eqs)
elif mode == 'validation':
  validate_models(model_path, splits, model_on_all_data_only, max_fold,
                  remove_incomplete_eqs)
elif mode == 'train_validate':
  train_models(custom_model, model_path, splits, hyperpars, overwrite_train,
               model_on_all_data, model_on_all_data_only, max_fold,
               remove_incomplete_eqs)
  print("\n")
  validate_models(model_path, splits, model_on_all_data_only, max_fold,
                  remove_incomplete_eqs)
elif mode == 'valid_order':
  valid_order(model_path, splits[0], data_folder, hyperpars)
elif mode == 'test_order':
  test_order(model_path, splits[0], data_folder, hyperpars)