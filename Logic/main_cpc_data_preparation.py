import warnings
with warnings.catch_warnings():
  warnings.simplefilter('ignore', RuntimeWarning)
  import models
  import utils
  
import numpy as np
import pandas as pd
from keras.models import load_model

data_folder = '/media/tom/cbd_drive/Kaggle/LANL/Data/'
encoder_path = '/home/tom/Kaggle/LANL/Models/initial_cpc-0-all_train_encoder.h5'
chunk_size = 150000
subchunk_size = 1500
num_sub_per_chunk = int(chunk_size/subchunk_size)
max_considered_chunks = -1 # Negative means consider all

# Load the encoder model
encoder_model = load_model(encoder_path,
                           custom_objects={'Attention': models.Attention})

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
  
# Load the global variables TRAIN_FEATURES and TEST_FEATURES if they have not
# been loaded before.
if ((not 'TRAIN_FEATURES' in locals()) and (
     not 'TRAIN_FEATURES' in globals())) or (
    (not 'TEST_FEATURES' in locals()) and (
     not 'TEST_FEATURES' in globals())):
  TRAIN_FEATURES = pd.read_csv(
      data_folder + 'train_features_scaled_keep_incomplete_target_quantile.csv')
  TEST_FEATURES = pd.read_csv(
      data_folder + 'test_features_scaled_keep_incomplete_target_quantile.csv')

for source in ['test', 'train']:
  # Read the signal and feature data
  signal_string = source.upper() + '_AUGMENT'
  feature_data = TRAIN_FEATURES if source == 'train' else TEST_FEATURES
  
  # Determine the rows of signal data to generate features for
  if source == 'test':
    num_test_files = feature_data.shape[0]
    feature_rows = np.arange(num_test_files)
    start_ids = chunk_size * feature_rows
  else:
    feature_rows = np.where(np.logical_and(
        feature_data.notrain_no_overlap_chunk.values,
        np.append(np.diff(feature_data.notrain_target_original) < 0,
                  np.array([False]))))[0]
    start_ids = feature_data.notrain_start_row.values[feature_rows]
  
  # Compute embeddings for the signal data
  raw_data = utils.get_sub_chunks_from_ids(locals()[signal_string],
                                           start_ids, chunk_size)
  if max_considered_chunks > 0:
    raw_data = raw_data[:max_considered_chunks]
    feature_rows = feature_rows[:max_considered_chunks]
    start_ids = start_ids[:max_considered_chunks]
  num_chunks = feature_rows.shape[0]
  s = raw_data.shape
  raw_data = [raw_data.reshape([-1, subchunk_size, s[-1]])]
  encodings = encoder_model.predict(raw_data, verbose=1)
  enc_columns = ['enc_' + str(i) for i in range(encodings.shape[1])]
  encodings_df = pd.DataFrame(data=encodings,
                              index=np.arange(encodings.shape[0]),
                              columns=enc_columns).astype(float)
  
  # Create a data frame of the notrain features
  segment_ids = np.repeat(feature_data.notrain_seg_id.values[feature_rows],
                          num_sub_per_chunk)
  start_rows = np.tile(subchunk_size*np.arange(num_sub_per_chunk), num_chunks)
  if source == 'test':
    targets = -999
    targets_original = -999
    eq_ids = -999
  else:
    start_rows += np.repeat(start_ids, num_sub_per_chunk)
    targets = np.interp(np.arange(0, num_chunks, 1/num_sub_per_chunk),
                        np.arange(num_chunks),
                        feature_data.target.values[feature_rows])
    targets_original = locals()[signal_string].time_to_failure.values[
        start_rows+subchunk_size-1]
    eq_ids = np.repeat(feature_data.notrain_eq_id.values[feature_rows],
                       num_sub_per_chunk)
  notrain_data = utils.ordered_dict(
      ['target', 'notrain_target_original', 'notrain_seg_id', 'notrain_eq_id',
       'notrain_start_row'],
      [targets, targets_original, segment_ids, eq_ids, start_rows])
  notrain_df = pd.DataFrame.from_dict(notrain_data)
  
  # Reshape the relevant feature data
  sub_col_ids = np.array([
      i for (i, v) in enumerate(feature_data.columns) if v[:4] == 'sub_'])
  sub_feat_count = int(len(sub_col_ids)/num_sub_per_chunk)
  sub_feature_names = feature_data.columns[sub_col_ids][
      np.arange(sub_feat_count)*num_sub_per_chunk]
  sub_feature_names = [f[:-11] for f in sub_feature_names]
  sub_features = feature_data.iloc[feature_rows, sub_col_ids].values
  sub_features = sub_features.reshape(
      [feature_rows.size, -1, num_sub_per_chunk])
  sub_features = np.transpose(sub_features, [0, 2, 1])
  sub_features = sub_features.reshape([encodings.shape[0], -1])
  sub_features_df = pd.DataFrame(data=sub_features,
                                 index=np.arange(encodings.shape[0]),
                                 columns=sub_feature_names)
  
  # Combine the embeddings with the feature data and include relevant notrain
  # columns (target, segment id, eq id and start row)
  combined = pd.concat([notrain_df, encodings_df, sub_features_df], axis=1)
  
  # Store the combined data
  data_path = data_folder + source + '_main_cpc_encodings_and_features.csv'
  combined.to_csv(data_path, index=False)
