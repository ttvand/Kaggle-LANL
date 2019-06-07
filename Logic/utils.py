import keras
from keras import backend as K
from collections import OrderedDict
from functools import partial, update_wrapper
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import tensorflow as tf
import threading

data_folder = '/media/tom/cbd_drive/Kaggle/LANL/Data/'


# Create an ordered dict from a list of keys and values
def ordered_dict(keys, vals):
  od = OrderedDict()
  for (k, v) in zip(keys, vals):
    od[k] = v
  
  return od

# Reshape features in order to include a time dimension for the recurrent model
def reshape_time_dim(features, targets, train_ids):
  raw_features = features.iloc[train_ids]
  sub_step_cols = [c for c in raw_features.columns if '_sub_step_' in c]
  num_steps = int(sub_step_cols[-1].split('_sub_step_')[-1])+1
  num_features = int(len(sub_step_cols) / num_steps)
  
  out = raw_features[sub_step_cols].values
  x_out = out.reshape([-1, num_features, num_steps]).transpose([0, 2, 1])
  y_out = np.tile(np.expand_dims(targets[train_ids], -1), [1, num_steps])
  return (x_out, y_out)


# Custom mae loss function that ignores the initial part of the predictions
def mae_last_part_loss(y_true, y_pred, drop_target_part=-1):
  time_steps = y_pred.shape.as_list()[1]
  cutoff_time_id = int(drop_target_part*time_steps)
  errors = y_true[:, cutoff_time_id:] - y_pred[:, cutoff_time_id:]
  
  return K.mean(K.abs(errors))


# Custom cross-entropy loss function that ignores the extreme parts of the
# predictions
def xe_central_part_loss(y_true, y_pred, drop_extreme_part=-1):
  time_steps = y_pred.shape.as_list()[1]
  cutoff_time_id = int(drop_extreme_part*time_steps)
  y_true_loss = y_true[:, cutoff_time_id:-cutoff_time_id]
  y_pred_loss = y_pred[:, cutoff_time_id:-cutoff_time_id]
  
  return K.mean(K.binary_crossentropy(y_true_loss, y_pred_loss))


# Compute the average predicted probabilities for two types of new data points:
# Actual gap points and no gap points
def gap_pred_statistics(gaps, preds, drop_extreme_part):
  time_steps = gaps.shape[1]
  cutoff_time_id = int(drop_extreme_part*time_steps)
  valid_gaps = gaps[:,  cutoff_time_id:-cutoff_time_id]
  valid_preds = preds[:,  cutoff_time_id:-cutoff_time_id]
  num_gaps = valid_gaps.sum()
  gap_av_pred = (valid_gaps*valid_preds).sum()/num_gaps
  nogap_av_pred = ((1-valid_gaps)*valid_preds).sum()/(valid_gaps.size-num_gaps)
  return (gap_av_pred, nogap_av_pred)


# Compute the prediction ratio of actual gap points to no gap points
def get_gap_predict_ratio(gaps, preds, drop_extreme_part):
  gap_pred_stats = gap_pred_statistics(gaps, preds, drop_extreme_part)

  return np.mean(gap_pred_stats[0]/gap_pred_stats[1])


# Tensorflow custom metric for the ratio of actual gap points to no gap points
def gap_predict_ratio(label, pred, drop_extreme_part=-1):
  return tf.py_func(get_gap_predict_ratio, [label, pred, drop_extreme_part],
                    tf.float64)


# Generator for the training of the gap model - sample uniform valid starting
# points ad infinitum
def generator_gap_batch(gap_data, ranges, hyperpars):
  batch_size = hyperpars['batch_size']
  block_steps = hyperpars['block_steps']
  max_gap_shifts = hyperpars['max_gap_shifts']
  num_ranges = ranges[0].size
  range_probs = (ranges[1] - ranges[0])/((ranges[1] - ranges[0]).sum())
  
  while True:
    # Assign the range ids
    range_ids = np.random.choice(num_ranges, size=batch_size, p=range_probs)
  
    # Initialize data matrices and targets
    acoustic_data = np.zeros((batch_size, block_steps, 1))
    targets = np.zeros((batch_size, block_steps))
    
    # Assign the appropriate data chunk
    for i, range_id in enumerate(range_ids):
      start_row = np.random.randint(ranges[0][range_id],
                                    ranges[1][range_id]-block_steps)
      
      # Shift the start row so that it always includes a gap - make sure to
      # avoid going outside of the given ranges!
      gap_targets = gap_data.is_gap.values[start_row:(start_row+block_steps)]
      if not gap_targets.sum():
        valid_directions = np.array([-1, 1])[np.array([
            start_row >= (ranges[0][range_id] + max_gap_shifts*block_steps),
            start_row < (ranges[1][range_id] - max_gap_shifts*block_steps)])]
        direction = np.random.choice(valid_directions)
        for shift_id in range(max_gap_shifts):
          start_row = start_row + direction*block_steps
          gap_targets = gap_data.is_gap.values[
              start_row:(start_row+block_steps)]
          if gap_targets.sum():
            break
          
      acoustic_data[i, :, 0] = gap_data.acoustic_data.values[
          start_row:(start_row+block_steps)]
      targets[i] = gap_targets
        
    yield acoustic_data, targets
    
    
# Generate overlapping feature matrices for all ranges so that the predictions
# Don't have predictions that the model was not trained on (except for the
# extremes of the ranges)
def get_gap_prediction_features(gap_data, ranges, hyperpars, order_start_rows,
                                max_considered_start_rows=float('inf')):
  block_steps = hyperpars['block_steps']
  block_increment = block_steps*(1-2*hyperpars['drop_extreme_part_loss_frac'])
  num_ranges = ranges[0].size
  start_rows = []
  
  for range_id in range(num_ranges):
    range_length = ranges[1][range_id] - ranges[0][range_id]
    num_start_rows_range = int(np.floor(range_length/block_increment)) + 1
    start_rows_range = (ranges[0][range_id] + np.arange(
        num_start_rows_range)*block_increment).astype(np.int64)
    start_rows_range = np.append(start_rows_range, ranges[1][range_id])
    start_rows.extend(start_rows_range.tolist())
  
  start_rows = np.array(start_rows)
  considered_start_rows = min(max_considered_start_rows, start_rows.size)
  start_rows = np.random.choice(
      start_rows, considered_start_rows, replace=False)
  if order_start_rows:
    start_rows = np.sort(start_rows)
  
  # Initialize data matrices and targets
  acoustic_data = np.zeros((considered_start_rows, block_steps, 1))
  targets = np.zeros((considered_start_rows, block_steps))
  
  # Assign the appropriate data chunk
  for i, start_row in enumerate(start_rows):
    acoustic_data[i, :, 0] = gap_data.acoustic_data.values[
        start_row:(start_row+block_steps)]
    targets[i] = gap_data.is_gap.values[start_row:(start_row+block_steps)]
        
  return acoustic_data, targets, start_rows


# Align the predictions from 'get_gap_prediction_features' to make sure that
# the predictions are in line with the training setup (except for the extremes)
# of the chunks
def align_test_gap_preds(preds, chunk_steps, start_rows, hyperpars,
                         file_names=None):
  # Get the ids of the burn-in predictions
  block_steps = hyperpars['block_steps']
  block_increment = start_rows[1]
  block_overlap = block_steps-block_increment
  half_block_overlap = int(block_overlap/2)
  num_pred_chunks_per_file = int(np.ceil(chunk_steps/block_increment))
  num_files = int(preds.shape[0]/num_pred_chunks_per_file)
  chunk_pred_count = num_pred_chunks_per_file*block_steps
  valid_pred_mask_chunk = np.zeros((chunk_pred_count), dtype=np.bool)
  for i in range(num_pred_chunks_per_file):
    if i == 0:
      mask_vals = [np.ones((block_steps-half_block_overlap)),
                   np.zeros((half_block_overlap))]
    elif i == (num_pred_chunks_per_file-2):
      valid_preds_sec_last = (
          chunk_steps - start_rows[num_pred_chunks_per_file-2] - block_steps)
      mask_vals = [np.zeros((half_block_overlap)),
                   np.ones((valid_preds_sec_last)),
                   np.zeros((
                       block_steps-half_block_overlap-valid_preds_sec_last))]
    elif i == (num_pred_chunks_per_file-1):
      mask_vals = [np.zeros((half_block_overlap)),
                   np.ones((block_steps-half_block_overlap))]
    else:
      mask_vals = [np.zeros((half_block_overlap)),
                   np.ones((block_steps-block_overlap)),
                   np.zeros((half_block_overlap))]
    mask_vals = np.concatenate(mask_vals).astype(np.bool)
    valid_pred_mask_chunk[i*block_steps:(i+1)*block_steps] = mask_vals
  
  # Assign the non burn-in predictions to the aligned output
  aligned_preds = np.zeros((chunk_steps, num_files))
  flat_preds = preds.flatten()
  for i in range(num_files):
    aligned_preds[:, i] = flat_preds[
        i*chunk_pred_count:(i+1)*chunk_pred_count][valid_pred_mask_chunk]
  
  # Prepare the aligned output data frame
  submission = pd.read_csv(data_folder + 'sample_submission.csv')
  if file_names is None:
    file_names = submission.seg_id.values[:num_files]
  aligned = pd.DataFrame(aligned_preds, columns=file_names)
  
  return aligned


# Helper function to generate pairs and labels for a given group and batch size
def get_labels_groups(batch_size, skip_group_size, lookup_ids=None,
                      sample_size_multiplier=1, num_decoys=-1):
  first, second = np.meshgrid(np.arange(batch_size), np.arange(batch_size))
  all_pairs = np.stack([second.flatten(), first.flatten()], axis=1)
  all_pairs = all_pairs[all_pairs[:, 0] != all_pairs[:, 1]]
  
  # Drop pairs from the same group size
  all_pairs = all_pairs[np.floor(all_pairs[:, 0]/skip_group_size) != np.floor(
      all_pairs[:, 1]/skip_group_size)]
      
  if lookup_ids is None:
    lookup_ids = np.arange(batch_size/skip_group_size/2)
  pair_ids = lookup_ids[
      np.floor(all_pairs/(skip_group_size*2)).astype(np.int8)]
  same_group_ids = pair_ids[:, 0] == pair_ids[:, 1]
  sample_size = int(batch_size/2*sample_size_multiplier)
  same_ids = np.where(np.logical_and(
      same_group_ids, all_pairs[:, 0] < all_pairs[:, 1]))[0]
  diff_ids = np.where(~same_group_ids)[0]
  
  if (same_ids.size >= sample_size) and (
      (diff_ids.size >= sample_size*num_decoys)):
    compare_ids = np.zeros((sample_size, (2+num_decoys)), dtype=int)
    compare_ids[:, :2] = all_pairs[
        np.random.choice(same_ids, sample_size, replace=False)]
    
    for i in range(sample_size):
      neg_options = all_pairs[np.logical_and(
          ~same_group_ids, all_pairs[:, 0] == compare_ids[i, 0])][:, 1]
      compare_ids[i, 2:] = np.random.choice(neg_options, size=num_decoys,
                 replace=False)
    
    compare_labels = np.zeros((1, sample_size, num_decoys+1), dtype=bool)
    compare_labels[:, :, 0] = True
  else:
    return None, None
    
  return np.expand_dims(compare_ids, 0), compare_labels
    

def get_start_ranges_train(ranges, batch_size, block_steps, train_eq_ids,
                           num_decoys, chunk_size=150000):
  # Select the earthquake ids by randomly sampling in the train ranges
  num_ranges = ranges[0].size
  range_probs = (ranges[1] - ranges[0])/((ranges[1] - ranges[0]).sum())
  eighth_batch_size = int(batch_size/8)
  quarter_batch_size = int(batch_size/4)
  range_ids = np.random.choice(num_ranges, size=eighth_batch_size,
                               p=range_probs)
  range_starts = ranges[0][range_ids]
  range_sizes =  ranges[1][range_ids]-range_starts
  range_samples = range_starts + (range_sizes*np.random.uniform(
      size=eighth_batch_size)).astype(np.int64)
  eq_ids = (np.tile(range_samples, (train_eq_ids.size, 1)) > (
     np.transpose(np.tile(train_eq_ids, (eighth_batch_size, 1))))).sum(0)-1
      
  # Sample random chunk starts within the same earthquake
  range_sizes = train_eq_ids[eq_ids+1] - train_eq_ids[eq_ids] - chunk_size
  chunk_range_starts = np.repeat(train_eq_ids[eq_ids], 2)
  chunk_range_sizes = np.repeat(range_sizes, 2)
  chunk_start_ranges = chunk_range_starts + (
      chunk_range_sizes*np.random.uniform(size=quarter_batch_size)).astype(
          np.int64)
  chunk_ranges = (chunk_start_ranges, chunk_start_ranges+chunk_size)
  start_ranges, compare_chunk_ids, compare_chunk_labels = get_start_ranges(
      chunk_ranges, batch_size, block_steps, num_decoys, use_each_range=True)
  
  # Select the ids to compare if the ranges come from the same earthquake.
  compare_eq_ids, compare_eq_labels = get_labels_groups(
      batch_size, 4, eq_ids, sample_size_multiplier=2, num_decoys=num_decoys)
  if compare_eq_ids is None:
    # Recomputing train start ranges - all eq ids identical.
    return get_start_ranges_train(ranges, batch_size, block_steps,
                                  train_eq_ids, num_decoys)
    
  return (start_ranges, compare_chunk_ids, compare_chunk_labels,
          compare_eq_ids, compare_eq_labels)
  

# Always generate "pairs of pairs" of start ranges for the CPC-like auxiliary
# task of predicting if the next sub-chunk is a continuation of the first or if
# two sub-chunks come from the same chunk.
# The sampling strategy is biased (central observations have a higher
# probability of being sampled), but this is fine for the purpose of the
# learned encoder (domain agnostic representation that captures local and
# global structure and is able to model the time to failure).
def get_start_ranges(ranges, batch_size, block_steps, num_decoys,
                     random_seq_jump=4096, chunk_size=150000,
                     use_each_range=False):
  num_ranges = ranges[0].size
  range_probs = (ranges[1] - ranges[0])/((ranges[1] - ranges[0]).sum())
  
  # Assign the range ids if they are not predetermined
  quarter_batch_size = int(batch_size/4)
  if use_each_range:
    assert quarter_batch_size == ranges[0].size
    range_ids = np.arange(quarter_batch_size)
  else:
    range_ids = np.random.choice(num_ranges, size=quarter_batch_size,
                                 p=range_probs)
  
  # Assign the appropriate data chunk
  random_jumps_1 = np.random.randint(random_seq_jump, size=quarter_batch_size)
  random_big_jumps = np.random.randint(int(chunk_size/4), int(chunk_size*3/4),
                                       size=quarter_batch_size)
  random_jumps_2 = np.random.randint(random_seq_jump, size=quarter_batch_size)
  range_starts = ranges[0][range_ids]
  range_sizes =  ranges[1][range_ids]-((4*block_steps) + random_jumps_1 + (
      random_big_jumps + random_jumps_2 + range_starts))
  first_start_ids = range_starts + (
      range_sizes*np.random.uniform(size=quarter_batch_size)).astype(np.int64)
  second_start_ids = first_start_ids + block_steps + random_jumps_1
  third_start_ids = second_start_ids + block_steps + random_big_jumps
  fourth_start_ids = third_start_ids + block_steps + random_jumps_2
  
  combined_ids = np.hstack((first_start_ids.reshape([-1, 1]),
                            second_start_ids.reshape([-1, 1]),
                            third_start_ids.reshape([-1, 1]),
                            fourth_start_ids.reshape([-1, 1])))
  start_ranges = combined_ids.flatten()
  
  # Compute the ids and labels for judging if two sub-chunks come from the same
  # chunk.
  chunk_comp_ids, chunk_comp_labels = get_labels_groups(
      batch_size, 2, num_decoys=num_decoys)
  
  return start_ranges, chunk_comp_ids, chunk_comp_labels


def get_sub_chunks_from_ids(data, start_ids, block_steps):
  batch_size = start_ids.size
  row_ids = np.repeat(start_ids, block_steps) + np.tile(
      np.arange(block_steps), batch_size)
  chunk_data = data.iloc[row_ids]
  if 'acoustic_data' in chunk_data.columns:
    chunk_values = chunk_data[['acoustic_data', 'gap_log_prediction']].values
  else:
    val_cols = [c for c in chunk_data.columns if c != 'target' and not 'notrain' in c]
    chunk_values = chunk_data[val_cols].values
  
  return chunk_values.reshape([batch_size, block_steps, -1])


# Generator for the training of the cpc encoder - sample uniform valid starting
# points ad infinitum
def generator_cpc_batch(train_data, test_data, train_ranges, train_eq_ids,
                        hyperpars, test_chunk_size=150000):
  batch_size = hyperpars['batch_size']
  block_steps = hyperpars['block_steps']
  num_decoys = hyperpars['num_decoys']
  num_test_files = int(test_data.shape[0] / test_chunk_size)
  test_ranges = (test_chunk_size * np.arange(num_test_files),
                 test_chunk_size * (1+np.arange(num_test_files)))
  
  while True:
    (train_start_ids, train_chunk_comp_ids, train_chunk_comp_labels,
     train_eq_comp_ids, train_eq_comp_labels) = get_start_ranges_train(
         train_ranges, batch_size, block_steps, train_eq_ids, num_decoys)
    (test_start_ids, test_chunk_comp_ids, test_chunk_comp_labels) = (
        get_start_ranges(test_ranges, batch_size, block_steps, num_decoys))
    train_batch_data = get_sub_chunks_from_ids(train_data, train_start_ids,
                                               block_steps)
    test_batch_data = get_sub_chunks_from_ids(test_data, test_start_ids,
                                              block_steps)
    mae_targets = train_data.time_to_failure.values[
        train_start_ids + block_steps - 1]
    domain_target_istrain = np.zeros((1, 2*batch_size), dtype=bool)
    domain_target_istrain[0, :batch_size] = True
    train_subchunk_comp_ids, train_subchunk_comp_labels = (
        get_labels_groups(batch_size, 1, num_decoys=num_decoys))
    test_subchunk_comp_ids, test_subchunk_comp_labels = (
        get_labels_groups(batch_size, 1, num_decoys=num_decoys))
    
    # Add a fixed batch dimension of 1 - we use a custom batch dimension since
    # the different sub-chunks are not handled independently because of the
    # CPC losses
    inputs = [np.expand_dims(train_batch_data, 0),
              np.expand_dims(test_batch_data, 0),
              
              train_subchunk_comp_ids,
              train_chunk_comp_ids,
              train_eq_comp_ids,
              
              test_subchunk_comp_ids,
              test_chunk_comp_ids,
              ]
    targets = [np.expand_dims(mae_targets, 0),
               domain_target_istrain,
               domain_target_istrain,
               
               train_subchunk_comp_labels,
               train_chunk_comp_labels,
               train_eq_comp_labels,
               
               test_subchunk_comp_labels,
               test_chunk_comp_labels,
               ]
    
    yield inputs, targets
    
    
# Generator for the test chunk order predictions of the base cpc encoder.
def generator_cpc_batch_test(test_data, hyperpars, encoder_model,
                             first_test_id):
  batch_size = hyperpars['batch_size']
  num_decoys = hyperpars['num_decoys']
  block_steps = hyperpars['block_steps']
  test_offset_multiplier = hyperpars['test_offset_multiplier']
  test_chunk_size = 150000
  overlap_steps = test_chunk_size - block_steps
  num_test_files = int(test_data.shape[0] / test_chunk_size)
  preds_per_chunk = hyperpars['test_predictions_per_chunk']
  test_ranges = (test_chunk_size * np.arange(num_test_files),
                 test_chunk_size * (1+np.arange(num_test_files)))
  second_test_id = 0
  
  chunk_offsets = []
  offset_sum = 0
  offset_id = 0
  for i in range(preds_per_chunk):
    chunk_offsets.append((offset_id, offset_sum-offset_id))
    if offset_id == 0:
      offset_sum += 1
      offset_id = offset_sum
    else:
      offset_id -= 1
  
  while True:
    test_start_ids = np.zeros((batch_size), dtype=int)
    test_chunk_comp_ids = np.ones((1, int(batch_size/2), num_decoys+2),
                                  dtype=int) * (batch_size-1)
    
    # Add the test start ids: at the end of the first and the beginning of the
    # second
    max_offset = np.array(chunk_offsets).max()
    for i in range(batch_size):
      offset = (i % (max_offset+1))*test_offset_multiplier
      second_offset = min((i // (max_offset+1)),
                          num_test_files-second_test_id)
      if i > max_offset:
        start_pos = test_ranges[0][second_test_id + second_offset - 1] + offset
      else:
        start_pos = test_ranges[0][first_test_id] + (
            overlap_steps - offset)
      test_start_ids[i] = start_pos
    
    # Assign the chunk comparison ids.
    num_second_ids = int(batch_size/2/(max_offset+1))
    test_chunk_comp_ids[0, :, 0] = np.tile(np.arange(max_offset+1),
                       num_second_ids)
    for second_offset in range(num_second_ids):
      second_offset_ids = second_offset*(max_offset+1)
      test_chunk_comp_ids[0,
                          second_offset_ids:(second_offset_ids+(max_offset+1)),
                          1:(1+max_offset+1)] = np.reshape(np.tile(
                          max_offset+1 + np.arange(max_offset+1) + (
                              second_offset_ids), max_offset+1),
      [-1, max_offset+1])
      
    test_batch_data = get_sub_chunks_from_ids(test_data, test_start_ids,
                                              block_steps)
    
    # Increment the first and second test ids
    second_test_id += num_second_ids
    if second_test_id >= num_test_files:
      second_test_id = 0
      first_test_id += 1
    
    # Add a fixed batch dimension of 1 - we use a custom batch dimension since
    # the different sub-chunks are not handled independently because of the
    # CPC losses
    inputs = [np.expand_dims(test_batch_data, 0),
              np.expand_dims(test_batch_data, 0),
              
              test_chunk_comp_ids,
              test_chunk_comp_ids,
              np.tile(test_chunk_comp_ids, [1, 2, 1]),
              test_chunk_comp_ids,
              test_chunk_comp_ids,
              ]
    
    yield inputs, []
    
    
def encode_inputs_model(data, model, block_steps):
  data_shape = list(data.shape)
  model_inputs = [np.reshape(data, [-1, block_steps] + data_shape[-1:])]
  encodings = model.predict(model_inputs, verbose=1)
  return np.reshape(encodings, data_shape[:2] + [-1, encodings.shape[-1]])
  
    
# Get the start ranges for the main CPC model: next chunk order prediction for
# train and domain adversarial models
def get_start_ranges_main(ranges, batch_size, block_steps, num_decoys,
                          random_seq_jump, subsequent_sampling=True):
  num_ranges = ranges[0].size
  range_probs = (ranges[1] - ranges[0])/((ranges[1] - ranges[0]).sum())
  
  # Assign the range ids if they are not predetermined
  sample_size = int(batch_size/2) if subsequent_sampling else batch_size
  range_ids = np.random.choice(num_ranges, size=sample_size, p=range_probs)
  
  # Assign the appropriate data chunk
  random_jumps = np.random.randint(random_seq_jump, size=sample_size)
  range_starts = ranges[0][range_ids]
  range_sizes =  ranges[1][range_ids]-(block_steps + range_starts + (
      int(subsequent_sampling)* (block_steps + random_jumps)))
  
  first_start_ids = range_starts + (
      range_sizes*np.random.uniform(size=sample_size)).astype(np.int64)
  second_start_ids = first_start_ids + block_steps + random_jumps
  
  if subsequent_sampling:
    combined_ids = np.hstack((first_start_ids.reshape([-1, 1]),
                              second_start_ids.reshape([-1, 1])))
    start_ranges = combined_ids.flatten()
  else:
    start_ranges = first_start_ids
  
  # Compute the ids and labels for judging if two sub-chunks come from the same
  # chunk.
  chunk_comp_ids, chunk_comp_labels = get_labels_groups(
      batch_size, 1, num_decoys=num_decoys)
  
  return start_ranges, chunk_comp_ids, chunk_comp_labels


# Generator for the training of the cpc encoder - sample uniform valid starting
# points ad infinitum
def generator_cpc_main_batch(train_data, test_data, train_ranges, hyperpars,
                             encoder_model):
  batch_size = hyperpars['batch_size']
  chunk_blocks = hyperpars['chunk_blocks']
  num_decoys = hyperpars['num_decoys']
  test_chunk_size = int(150000/hyperpars['block_steps'])
  num_test_files = int(test_data.shape[0] / test_chunk_size)
  test_ranges = (test_chunk_size * np.arange(num_test_files),
                 test_chunk_size * (1+np.arange(num_test_files)))
  
  while True:
    (train_start_ids, train_chunk_comp_ids,
     train_chunk_comp_labels) = get_start_ranges_main(
         train_ranges, batch_size, chunk_blocks, num_decoys,
         random_seq_jump=hyperpars['random_seq_jump'])
    (test_start_ids, _, _) = get_start_ranges_main(
        test_ranges, batch_size, chunk_blocks, num_decoys,
        random_seq_jump=hyperpars['random_seq_jump'],
        subsequent_sampling=False)
    train_batch_data = get_sub_chunks_from_ids(train_data, train_start_ids,
                                               chunk_blocks)
    test_batch_data = get_sub_chunks_from_ids(test_data, test_start_ids,
                                              chunk_blocks)
    mae_targets = train_data.target.values[
        train_start_ids + chunk_blocks - 1]
    domain_target_istrain = np.zeros((1, 2*batch_size), dtype=bool)
    domain_target_istrain[0, :batch_size] = True
    
    # Add a fixed batch dimension of 1 - we use a custom batch dimension since
    # the different sub-chunks are not handled independently because of the
    # CPC losses
    inputs = [np.expand_dims(train_batch_data, 0),
              np.expand_dims(test_batch_data, 0),
              train_chunk_comp_ids,
              ]
    targets = [np.expand_dims(mae_targets, 0),
               domain_target_istrain,
               domain_target_istrain,
               train_chunk_comp_labels,
               ]
    
    yield inputs, targets
    
    
class generator_cpc_main_batch_thread_safe:
  def __init__(self, train_data, test_data, train_ranges, hyperpars,
               encoder_model):
    self.train_data = train_data
    self.test_data = test_data
    self.train_ranges = train_ranges
    self.hyperpars = hyperpars
    self.encoder_model = encoder_model
    test_chunk_size = int(150000/hyperpars['block_steps'])
    self.test_chunk_size = test_chunk_size
    
    self.batch_size = hyperpars['batch_size']
    self.chunk_blocks = hyperpars['chunk_blocks']
    self.num_decoys = hyperpars['num_decoys']
    num_test_files = int(test_data.shape[0] / test_chunk_size)
    self.test_ranges = (test_chunk_size * np.arange(num_test_files),
                        test_chunk_size * (1+np.arange(num_test_files)))
    
    self.lock = threading.Lock()

  def __iter__(self):
    return self

  def __next__(self):
    with self.lock:
      (train_start_ids, train_chunk_comp_ids,
       train_chunk_comp_labels) = get_start_ranges_main(
           self.train_ranges, self.batch_size, self.chunk_blocks,
           self.num_decoys, random_seq_jump=self.hyperpars['random_seq_jump'])
      (test_start_ids, _, _) = get_start_ranges_main(
          self.test_ranges, self.batch_size, self.chunk_blocks, self.num_decoys,
          random_seq_jump=self.hyperpars['random_seq_jump'],
          subsequent_sampling=False)
      train_batch_data = get_sub_chunks_from_ids(
          self.train_data, train_start_ids, self.chunk_blocks)
      test_batch_data = get_sub_chunks_from_ids(
          self.test_data, test_start_ids, self.chunk_blocks)
      mae_targets = self.train_data.target.values[
          train_start_ids + self.chunk_blocks - 1]
      domain_target_istrain = np.zeros((1, 2*self.batch_size), dtype=bool)
      domain_target_istrain[0, :self.batch_size] = True
      
      # Add a fixed batch dimension of 1 - we use a custom batch dimension since
      # the different sub-chunks are not handled independently because of the
      # CPC losses
      inputs = [np.expand_dims(train_batch_data, 0),
                np.expand_dims(test_batch_data, 0),
                train_chunk_comp_ids,
                ]
      targets = [np.expand_dims(mae_targets, 0),
                 domain_target_istrain,
                 domain_target_istrain,
                 train_chunk_comp_labels,
                 ]
      
      
      return inputs, targets
    
    
# Generator for the test chunk order predictions of the main cpc encoder.
def generator_cpc_main_batch_test(test_data, hyperpars, encoder_model,
                                  first_test_id):
  batch_size = hyperpars['batch_size']
  chunk_blocks = hyperpars['chunk_blocks']
  num_decoys = hyperpars['num_decoys']
  block_steps = hyperpars['block_steps']
  test_chunk_size = int(150000/block_steps)
  overlap_block_subchunks = test_chunk_size-chunk_blocks
  num_test_files = int(test_data.shape[0] / test_chunk_size)
  preds_per_chunk = hyperpars['test_predictions_per_chunk']
  test_ranges = (test_chunk_size * np.arange(num_test_files),
                 test_chunk_size * (1+np.arange(num_test_files)))
  second_test_id = 0
  
  chunk_offsets = []
  offset_sum = 0
  offset_id = 0
  for i in range(preds_per_chunk):
    chunk_offsets.append((offset_id, offset_sum-offset_id))
    if offset_id == 0:
      offset_sum += 1
      offset_id = offset_sum
    else:
      offset_id -= 1
  
  while True:
#    import pdb; pdb.set_trace()
    test_start_ids = np.zeros((batch_size), dtype=int)
    test_chunk_comp_ids = np.ones((1, int(batch_size/2), num_decoys+2),
                                  dtype=int) * (batch_size-1)
    
    # Add the test start ids: at the end of the first and the beginning of the
    # second
    max_offset = np.array(chunk_offsets).max()
    for i in range(batch_size):
      offset = i % (max_offset+1)
      second_offset = min((i // (max_offset+1)),
                          num_test_files-second_test_id)
      if i > max_offset:
        start_pos = test_ranges[0][second_test_id + second_offset - 1] + offset
      else:
        start_pos = test_ranges[0][first_test_id] + (
            overlap_block_subchunks - offset)
      test_start_ids[i] = start_pos
    
    # Assign the chunk comparison ids.
    num_second_ids = int(batch_size/2/(max_offset+1))
    test_chunk_comp_ids[0, :, 0] = np.tile(np.arange(max_offset+1),
                       num_second_ids)
    for second_offset in range(num_second_ids):
      second_offset_ids = second_offset*(max_offset+1)
      test_chunk_comp_ids[0,
                          second_offset_ids:(second_offset_ids+(max_offset+1)),
                          1:(1+max_offset+1)] = np.reshape(np.tile(
                          max_offset+1 + np.arange(max_offset+1) + (
                              second_offset_ids), max_offset+1),
      [-1, max_offset+1])
#      position_counters = np.zeros((max_offset+1), dtype=int)
#      for i in range(preds_per_chunk):
#        left_offset = chunk_offsets[i][0]
#        right_offset = chunk_offsets[i][1]
#        row_id = left_offset + second_offset_ids
#        col_id = 1+position_counters[left_offset]
#        position_counters[left_offset] += 1
#        comp_id = max_offset+1 + second_offset_ids + right_offset
#        
#        test_chunk_comp_ids[0, row_id, col_id] = comp_id
      
    test_batch_data = get_sub_chunks_from_ids(test_data, test_start_ids,
                                              chunk_blocks)
    
    # Increment the first and second test ids
    second_test_id += num_second_ids
    if second_test_id >= num_test_files:
      second_test_id = 0
      first_test_id += 1
    
    # Add a fixed batch dimension of 1 - we use a custom batch dimension since
    # the different sub-chunks are not handled independently because of the
    # CPC losses
    inputs = [np.expand_dims(test_batch_data, 0),
              np.expand_dims(test_batch_data, 0),
              test_chunk_comp_ids,
              ]
    
    yield inputs, []
    
    
# Generator for the training of the sequential RNN
# Sample uniform valid starting points ad infinitum
def generator_rnn_sequential_batch(data, ranges, hyperpars, target_quantile):
  batch_size = hyperpars['batch_size']
  chunk_blocks = hyperpars['chunk_blocks']
  num_ranges = ranges[0].size
  range_probs = (ranges[1] - ranges[0])/((ranges[1] - ranges[0]).sum())
  
  # Extract the targets and features
  all_targets = data.target.values if target_quantile else (
      data.notrain_target_original.values)
  feature_cols = [
      c for c in data.columns if c != 'target' and not 'notrain' in c]
  num_features = 0
  hyperpars['input_dimension_freq']
  if not hyperpars['include_freq_features']:
    feature_cols = [c for c in feature_cols if c[:4] != 'sub_']
  else:
    num_features += hyperpars['input_dimension_freq']
  if not hyperpars['include_cpc_features']:
    feature_cols = [c for c in feature_cols if c[:4] != 'enc_']
  else:
    num_features += hyperpars['input_dimension_cpc']
  all_features = data[feature_cols].values
  
  while True:
    # Assign the range ids
    range_ids = np.random.choice(num_ranges, size=batch_size, p=range_probs)
  
    # Initialize data matrices and targets
    batch_features = np.zeros((batch_size, chunk_blocks, num_features))
    batch_targets = np.zeros((batch_size, chunk_blocks))
    start_rows = np.zeros((batch_size), dtype=np.int64)
    
    # Assign the appropriate data chunk
    for i, range_id in enumerate(range_ids):
      start_row = np.random.randint(ranges[0][range_id],
                                    ranges[1][range_id]-chunk_blocks)
      
      batch_features[i] = all_features[start_row:(start_row+chunk_blocks), :]
      batch_targets[i] = all_targets[start_row:(start_row+chunk_blocks)]
      start_rows[i] = start_row

    yield batch_features, batch_targets, start_rows
    
    
# Generate overlapping feature matrices for all ranges so that the predictions
# Don't have predictions that the model was not trained on (except for the
# extremes of the ranges)
def get_rnn_prediction_features(data, ranges, hyperpars, order_start_rows,
                                max_considered_start_rows=float('inf')):
  chunk_blocks = hyperpars['chunk_blocks']
  block_increment = chunk_blocks*(hyperpars['predict_last_part_frac'])
  num_ranges = ranges[0].size
  start_rows = []
  
  for range_id in range(num_ranges):
    range_length = ranges[1][range_id] - ranges[0][range_id]
    num_start_rows_range = int(np.floor(range_length/block_increment)) + 1
    start_rows_range = (ranges[0][range_id] + np.arange(
        num_start_rows_range)*block_increment).astype(np.int64)
    start_rows_range = np.append(start_rows_range, ranges[1][range_id])
    start_rows.extend(start_rows_range.tolist())
  
  start_rows = np.array(start_rows)
  considered_start_rows = min(max_considered_start_rows, start_rows.size)
  start_rows = np.random.choice(
      start_rows, considered_start_rows, replace=False)
  if order_start_rows:
    start_rows = np.sort(start_rows)
  
  # Extract the features
  feature_cols = [
      c for c in data.columns if c != 'target' and not 'notrain' in c]
  num_features = 0
  if not hyperpars['include_freq_features']:
    feature_cols = [c for c in feature_cols if c[:4] != 'sub_']
  else:
    num_features += hyperpars['input_dimension_freq']
  if not hyperpars['include_cpc_features']:
    feature_cols = [c for c in feature_cols if c[:4] != 'enc_']
  else:
    num_features += hyperpars['input_dimension_cpc']
  all_features = data[feature_cols].values
  
  # Initialize the features data array
  features = np.zeros((considered_start_rows, chunk_blocks, num_features))
  
  # Assign the appropriate data chunk
  for i, start_row in enumerate(start_rows):
    features[i] = all_features[start_row:(start_row+chunk_blocks)]
        
  return features, start_rows


# Custom Callback for checkpointing the encoder model only
# Inspired by https://stackoverflow.com/questions/50983008/how-to-save-best-weights-of-the-encoder-part-only-during-auto-encoder-training
# Callback source: https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L633
class EncoderCheckpointer(keras.callbacks.Callback):
  def __init__(self, filepath, encoder, monitor, mode, save_best_only,
               verbose=0):
    self.filepath = filepath
    self.encoder = encoder
    self.monitor = monitor
    self.save_best_only = save_best_only
    self.verbose = verbose
    
    self.monitor_op = np.less if mode == 'min' else np.greater
    self.best = np.Inf if mode == 'min' else -np.Inf
  
  def on_epoch_end(self, epoch, logs=None):
    current = logs.get(self.monitor)
    if not self.save_best_only or self.monitor_op(current, self.best):
      if self.verbose > 0:
        print('\nSaving the encoder model to {}'.format(self.filepath))
      self.best = current
      self.encoder.save(self.filepath, overwrite=True)
    
    
# Helper function to initialize a function with partial arguments while 
# preserving the structure of the object (required by Keras)
# Source: http://louistiao.me/posts/adding-__name__-and-__doc__-attributes-to-functoolspartial-objects/
def wrapped_partial(func, *args, **kwargs):
  partial_func = partial(func, *args, **kwargs)
  update_wrapper(partial_func, func)
  return partial_func
  

# Apply random initializations to make the results as reproducible as possible
def make_results_reproducible(K, seed):
  # NOTE: Multiple threads are a potential source of
  # non-reproducible results.
  # For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  random.seed(seed)
  
#  # For reproducibility at the cost of performance one should add:
#  # Working hypothesis Tom: call this after each K.clear_session
#  session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
#                                inter_op_parallelism_threads=1)
#  tf.set_random_seed(seed)
#  sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
#  K.set_session(sess)


# Custom Keras callback for plotting learning progress
class PlotLosses(keras.callbacks.Callback):
  def on_train_begin(self, logs={}):
    self.i = 0
    self.x = []
    self.val_losses = []
    self.fig = plt.figure()
    self.logs = []
    
    loss_extensions = [
#        '',
        'train_mae_prediction',
#        'domain_prediction',
#        'train_subchunk_predictions', 'train_chunk_predictions',
#        'train_eq_predictions', 'test_subchunk_predictions',
#        'test_chunk_predictions'
]
    
    self.best_loss_key = 'train_mae_prediction_loss'
    self.loss_keys = [e + ('_' if e else '') + 'loss' for e in loss_extensions]
    self.losses = {k: [] for k in self.loss_keys}

  def on_epoch_end(self, epoch, logs={}):
    self.logs.append(logs)
    self.x.append(self.i)
    for k in self.loss_keys:
      self.losses[k].append(logs.get(k))
    self.i += 1
    
    best_loss = np.repeat(np.array(self.losses[self.best_loss_key]).min(),
                              self.i).tolist()
    best_id = (1+np.repeat(
        np.array(self.losses[self.best_loss_key]).argmin(), 2)).tolist()
    for k in self.loss_keys:
      plt.plot([1+x for x in self.x], self.losses[k], label=k)
    all_losses = np.array(list(self.losses.values())).flatten()
    if len(self.losses) > 1:
      plt.plot([1+x for x in self.x], best_loss, linestyle='--', color='r',
               label='')
      plt.plot(best_id, [min(all_losses) - 0.1, best_loss[0]],
               linestyle='--', color='r', label='')
    plt.ylim(min(all_losses) - 0.1, max(all_losses) + 0.1)
    plt.legend()
    plt.show()