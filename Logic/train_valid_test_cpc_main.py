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

## Turn warnings into errors, great for debugging!
warnings_to_errors = False
import warnings
if warnings_to_errors:
  warnings.filterwarnings('error')
else:
  warnings.filterwarnings('default')

# Set the path to the data folder
data_folder = '/media/tom/cbd_drive/Kaggle/LANL/Data/'
#data_folder = '/home/tom/Kaggle/LANL/Data/'

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
 
# Compute the global variable TRAIN_EQ_IDS if it has not been loaded before
if not 'TRAIN_EQ_IDS' in locals() and not 'TRAIN_EQ_IDS' in globals():
  TRAIN_EQ_IDS = 1+np.where(
      np.diff(TRAIN_DATA.target.values) > 0)[0]
  TRAIN_EQ_IDS = np.insert(TRAIN_EQ_IDS, 0, 0)
  TRAIN_EQ_IDS = np.append(TRAIN_EQ_IDS, TRAIN_DATA.shape[0])

# Helper function for naming the model path given the fold id
def get_fold_description(fold, num_folds):
  return fold if fold < num_folds else 'all_train'

# Fit a selected model and fold
def fit_model(custom_model, save_path_orig, split, fold, num_folds,
              train_val_split, hyperpars, overwrite_train,
              remove_incomplete_eqs):
  fold_description = get_fold_description(fold, num_folds)
  save_path = '{}-{}-{}.h5'.format(save_path_orig, split, fold_description)
  if not os.path.exists(save_path) or overwrite_train:
    if fold == num_folds:
      eq_ids = np.arange(TRAIN_EQ_IDS.shape[0])
      if remove_incomplete_eqs:
        eq_ids = eq_ids[1:-1]
    else:
      start_eq_ids = TRAIN_DATA.notrain_eq_id.values[np.array([
          np.where(TRAIN_DATA.notrain_start_row.values == r)[0][0] for r in (
              train_val_split[split][fold][0][0])])]
      end_eq_ids = TRAIN_DATA.notrain_eq_id.values[np.array([
          np.where(TRAIN_DATA.notrain_start_row.values == r)[0][0] for r in (
              train_val_split[split][fold][0][1])])]
      eq_ids = np.array([i for (start, stop) in zip(
          start_eq_ids, end_eq_ids) for i in range(start, stop)])
    
    train_ranges = (TRAIN_EQ_IDS[eq_ids], TRAIN_EQ_IDS[eq_ids+1])
      
    (inputs, outputs, encoder_model) = custom_model(hyperpars, ENCODER_PATH)
    train_gen = utils.generator_cpc_main_batch(
        TRAIN_DATA, TEST_DATA, train_ranges, hyperpars, encoder_model)
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
    reduce_lr = ReduceLROnPlateau(factor=1/math.sqrt(10), verbose=1,
                                  patience=hyperpars['reduce_lr_patience'],
                                  min_lr=hyperpars['initial_lr']/100,
                                  monitor=monitor,
                                  mode=monitor_mode)
    callbacks = [earlystopper, checkpointer, reduce_lr]
    callbacks.append(utils.PlotLosses())
#    callbacks.append(TensorBoard('./Graph'))
    
    model.fit_generator(train_gen,
                        steps_per_epoch=hyperpars['steps_per_epoch'],
                        epochs=hyperpars['epochs'],
                        callbacks=callbacks,
                        validation_data=None,
                        workers=1,
                        use_multiprocessing=False,
                        )
    
  # Evaluate the trained model performance
  validate_model(save_path_orig, split, fold, num_folds, train_val_split,
                 hyperpars, hyperpars['train_valid_batch'],
                 remove_incomplete_eqs, encoder_model)

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
                   hyperpars, validation_batch, remove_incomplete_eqs,
                   encoder_model):
  if fold == num_folds:
    eq_ids = np.arange(TRAIN_EQ_IDS.shape[0])
    if remove_incomplete_eqs:
      eq_ids = eq_ids[1:-1]
  else:
    start_eq_ids = TRAIN_DATA.notrain_eq_id.values[np.array([
        np.where(TRAIN_DATA.notrain_start_row.values == r)[0][0] for r in (
            train_val_split[split][fold][1][0])])]
    end_eq_ids = TRAIN_DATA.notrain_eq_id.values[np.array([
        np.where(TRAIN_DATA.notrain_start_row.values == r)[0][0] for r in (
            train_val_split[split][fold][1][1])])]
    eq_ids = np.array([i for (start, stop) in zip(
        start_eq_ids, end_eq_ids) for i in range(start, stop)])
      
  valid_ranges = (TRAIN_EQ_IDS[eq_ids], TRAIN_EQ_IDS[eq_ids+1])
  
  valid_gen = utils.generator_cpc_main_batch(
      TRAIN_DATA, TEST_DATA, valid_ranges, hyperpars, encoder_model)
  
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
  model_inputs = []
  for i in range(len(data[0][0])):
    model_inputs.append(np.concatenate([d[0][i] for d in data]))
  preds = model.predict(model_inputs, verbose=1)
#  import pdb; pdb.set_trace()
#  print('Next chunk predictions: ({}, {})'.format(
#      preds[3][:,:,0].mean(), preds[3][:,:,1:].mean()))
  
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
  encoder_model = load_model(ENCODER_PATH,
                             custom_objects={'Attention': models.Attention})
  for split_id, split in enumerate(splits):
    print('Processing split {} of {}'.format(split_id+1, len(splits)))
    split_maes = []
    for fold in range(num_validation_models):
      if not model_on_all_data_only or fold == num_folds:
        print('Processing fold {} of {}'.format(fold+1, num_validation_models))
        fold_mae = validate_model(
            save_path, split, fold, num_folds, train_val_split, hyperpars,
            hyperpars['validation_valid_batch'], remove_incomplete_eqs,
            encoder_model)
        split_maes.append((fold_mae))
    split_mae = np.array(split_maes).mean()
    split_maes = [split_mae] + split_maes 
    print('Split OOF MAE: {0:.3f}'.format(split_mae))
    valid_maes.append((split_maes, splits[split_id]))
    
  return valid_maes


def valid_order(model_path, split, data_folder):
  train_val_split = preprocess.train_val_split_gaps()
  num_folds = len(train_val_split[0])
  num_train_models = 1
  
  # Set the validation data to the first eleven validation earthquake ids since
  # the first validation earthquake
  first_val_ranges = train_val_split[split][0][1] # First fold, validation
  other_train_features = preprocess.get_preprocessed(
      'train', remove_overlap_chunks=True)[1]
  first_eq_id = other_train_features.eq_id.values[np.where(
      other_train_features.start_row.values == first_val_ranges[0][0])[0][0]]
  eq_ids = TRAIN_DATA.notrain_eq_id.values
  valid_rows = np.where(np.logical_and(eq_ids >= first_eq_id,
                                       eq_ids < (first_eq_id + 11)))[0]
  VALID_DATA = TRAIN_DATA.iloc[valid_rows]
  num_valid_files = int(VALID_DATA.shape[0]/(150000/hyperpars['block_steps']))
  
  order_probs = np.zeros((num_train_models, num_valid_files, num_valid_files))
  
  for fold in range(num_train_models):
    print('\nProcessing fold {} of {}'.format(fold+1, num_train_models))
    K.clear_session()
    encoder_model = load_model(ENCODER_PATH, custom_objects={
        'Attention': models.Attention})
    comp_rows_per_it = 4
    
    fold_description = get_fold_description(fold, num_folds)
    fold_model_path = '{}-{}-{}.h5'.format(model_path, split, fold_description)
    model = load_model(fold_model_path, custom_objects={
              'Attention': models.Attention,
              'GradientReversal': models.GradientReversal,
              'tf': tf,
              })
    
    num_iterations = int(num_valid_files/comp_rows_per_it)
    for i in range(num_iterations):
      gc.collect()
      print('\nIteration {} of {}'.format(i+1, num_iterations))
      first_test_id = int(comp_rows_per_it*i)
      test_gen = utils.generator_cpc_main_batch_test(
          VALID_DATA, hyperpars, encoder_model, first_test_id=first_test_id)
    
      # Generate the test data by calling the generator *N* times
      N = int(num_valid_files/4*comp_rows_per_it)
      test_data = list(itertools.islice(test_gen, N))
      test_preds = make_predictions(model_path, split=-1, fold=-1,
                                    num_folds=-1, data=test_data, model=model)
      order_preds = test_preds[3][:, :, :4].mean(-1).reshape(
          [test_preds[3].shape[0], -1, 4]).mean(-1)
      order_probs[fold, first_test_id:(first_test_id+comp_rows_per_it)] = (
          order_preds.reshape([comp_rows_per_it, -1]))
      
    save_path = data_folder + 'valid_order_probs.npy'
    np.save(save_path, order_probs) # np.load(save_path)


def test_order(model_path, split, model_on_all_data, data_folder):
  train_val_split = preprocess.train_val_split_gaps()
  num_folds = len(train_val_split[0])
  num_train_models = num_folds + int(model_on_all_data)
  num_test_files = 2624
  order_probs = np.zeros((num_train_models, num_test_files, num_test_files))
  
  for fold in range(num_train_models):
    print('\nProcessing fold {} of {}'.format(fold+1, num_train_models))
    K.clear_session()
    encoder_model = load_model(ENCODER_PATH, custom_objects={
        'Attention': models.Attention})
    comp_rows_per_it = 16
    
    fold_description = get_fold_description(fold, num_folds)
    fold_model_path = '{}-{}-{}.h5'.format(model_path, split, fold_description)
    model = load_model(fold_model_path, custom_objects={
              'Attention': models.Attention,
              'GradientReversal': models.GradientReversal,
              'tf': tf,
              })
    
    num_iterations = int(num_test_files/comp_rows_per_it)
    for i in range(num_iterations):
      gc.collect()
      print('\nIteration {} of {}'.format(i+1, num_iterations))
      first_test_id = int(comp_rows_per_it*i)
      test_gen = utils.generator_cpc_main_batch_test(
          TEST_DATA, hyperpars, encoder_model, first_test_id=first_test_id)
    
      # Generate the test data by calling the generator *N* times
      N = int(num_test_files/4*comp_rows_per_it)
      test_data = list(itertools.islice(test_gen, N))
      test_preds = make_predictions(model_path, split=-1, fold=-1,
                                    num_folds=-1, data=test_data, model=model)
      order_preds = test_preds[3][:, :, :4].mean(-1).reshape(
          [test_preds[3].shape[0], -1, 4]).mean(-1)
      order_probs[fold, first_test_id:(first_test_id+comp_rows_per_it)] = (
          order_preds.reshape([comp_rows_per_it, -1]))
      
    save_path = data_folder + 'test_order_probs.npy'
    np.save(save_path, order_probs) # np.load(save_path)
      
    
##########################################
# Main logic for training and validation #
##########################################
mode = ['train', 'validation', 'train_validate', 'valid_order', 'test_order',
        'all_order', 'no_action'][3]
splits = [0]
model_description = 'main_cpc'
encoder_model_path = 'initial_cpc-0-all_train'
custom_model = models.main_cpc
overwrite_train = True
model_on_all_data = True
model_on_all_data_only = False
remove_incomplete_eqs = True
max_fold = float('inf')

# List the loss types for the different outputs
LOSS_TYPES = {
    'train_mae_prediction': 'mean_absolute_error',
    'domain_prediction': 'binary_crossentropy',
    'additional_domain_prediction': 'binary_crossentropy',
    'train_chunk_predictions': 'categorical_crossentropy',
    }

hyperpars = {
    'block_steps': 1500,
    'chunk_blocks': int(150000*0.8/1500),
    'input_dimension': 16+17,
    'epochs': 40,
    'steps_per_epoch': 1000,
    'batch_size': 32,
    'num_decoys': 4,
    
    'initial_lr': 1e-3,
    'es_patience': 15,
    'reduce_lr_patience': 4,
    'loss_weights': {
        'train_mae_prediction': 1.0,
        'domain_prediction': 0.5,
        'additional_domain_prediction': 0.5,
        'train_chunk_predictions': 1.0,
        },
    
    'use_attention_encoder': False,
    'grad_rev_lambda': 0.5, # -1 corresponds to no gradient reversal
    'encoder_input_embedding_layers': [],
    'encoder_output_embedding_layers': [],
    'encoder_recurrent_cells': [32, 32],
    'mae_prediction_layers': [32, 16],
    'domain_prediction_layers': [32, 16],
    'additional_domain_prediction_layers': [32, 16],
    'chunk_prediction_layers': [16, 16],
    'input_dropout': 0.3,
    'encoder_mlp_dropout': 0.25,
    'gru_dropout': 0.25,
    'prediction_dropout': 0.25,
    
    'train_valid_batch': 100,
    'validation_valid_batch': 2000,
    
    'random_seq_jump': 20,
    'test_predictions_per_chunk': 10,
}

model_path = '/home/tom/Kaggle/LANL/Models/' + model_description
encoder_path = '/home/tom/Kaggle/LANL/Models/' + encoder_model_path
ENCODER_PATH = encoder_path + '_encoder.h5'
      
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
  valid_order(model_path, splits[0], data_folder)
elif mode == 'test_order':
  test_order(model_path, splits[0], model_on_all_data, data_folder)
elif mode == 'all_order':
  test_order(model_path, splits[0], model_on_all_data, data_folder)
  valid_order(model_path, splits[0], data_folder)