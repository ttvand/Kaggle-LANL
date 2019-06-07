import numpy as np
import pandas as pd

data_folder = '/media/tom/cbd_drive/Kaggle/LANL/Data/'
mode = ['validation', 'test'][1]
pred_extension = '_test' if mode == 'test' else '_valid'
base_pred_path = data_folder + 'gap_model_aligned_predictions'
pred_path = base_pred_path + pred_extension + '.csv'
ttf_file = '../Submissions/19-03-05-08-55 - LightGBM initial baseline.csv'

# Run this line when changing the evaluation mode
#del gap_preds; del order_preds; del raw_order_preds; del pattern_summary_all

# Load the gap predictions for all chunks
if (not 'gap_preds' in locals()):
  gap_preds = pd.read_csv(pred_path)
  test_file_names = list(gap_preds.columns.values)
  gap_preds = gap_preds.values
  
# Load the order predictions for all chunks
if (not 'order_preds' in locals()):
  order_preds_path = data_folder + pred_extension[1:] + '_order_probs.npy'
  order_preds = np.load(order_preds_path)
  for i in range(order_preds.shape[0]):
    np.fill_diagonal(order_preds[i], 0)
  valid_fold_ids = np.sum(order_preds, (1, 2)) > 0
  order_preds_mean = order_preds[valid_fold_ids].mean(0)
  
# Load the high frequency order predictions for all chunks
if (not 'raw_order_preds' in locals()):
  raw_order_preds_path = data_folder + pred_extension[1:] + (
      '_order_probs_raw_signal.npy')
  raw_order_preds = np.load(raw_order_preds_path)
  np.fill_diagonal(raw_order_preds, 0)
  
# Load the global variables TRAIN_DATA and TEST_DATA if they have not
# been loaded before.
if ((not 'train_data' in locals()) and (
     not 'train_data' in globals())) or (
    (not 'test_data' in locals()) and (
     not 'test_data' in globals())):
  train_data = pd.read_csv(data_folder + 'train_features.csv')
  feature_rows = np.where(np.logical_and(
        train_data.notrain_no_overlap_chunk.values,
        np.append(np.diff(train_data.target) < 0, np.array([False]))))[0]
  train_data = train_data.iloc[feature_rows]
  test_data = pd.read_csv(data_folder + 'test_features.csv')
  chunk_std = test_data.chunk_std if mode == 'test' else train_data.chunk_std
  chunk_std = chunk_std.values

# Compute the argmax counts and average ranks of the different offsets
num_files = raw_order_preds.shape[0]
offsets = np.arange(-50, 50)
argmax_counts = np.zeros((offsets.size, 3))
sort_ids = np.argsort(-order_preds_mean, 1)
base_match_matrix_t = np.tile(np.arange(num_files), [num_files, 1])
base_match_matrix = np.transpose(base_match_matrix_t)
for i, offset in enumerate(offsets):
  argmax_counts[i, 0] = offset
  argmax_counts[i, 1] = (sort_ids[:, 0] == (
      offset+np.arange(order_preds_mean.shape[0]))).sum()
#  match_matrix = base_match_matrix + offset
#  match_ids = sort_ids == match_matrix
#  argmax_counts[i, 2] = base_match_matrix_t[match_ids].mean()
  
# List all possible gap patterns for the test chunks given the initial step
def pattern_summary_initial_step(gap_preds, initial_step=0, initial_cycle_id=0,
                                 cycle_4095=1280, num_chunks=2624,
                                 step_overrides=None, half_block_overlap=200):
  chunk_size = 150000
  patterns = []
  string_patterns = []
  first_occurences = []
  step = initial_step
  cycle_id = initial_cycle_id
  chunk_id = 0
  gap_ids = [initial_step]
  
  while chunk_id < num_chunks:
    prev_step = step
    gap_increment = 4096 - (cycle_id == 0)
    step = (step + gap_increment) % chunk_size
    if step < prev_step:
      string_pattern = '-'.join([str(g) for g in gap_ids])
      first_occurences.append(string_pattern not in string_patterns)
      patterns.append(np.array(gap_ids))
      string_patterns.append(string_pattern)
      chunk_id = chunk_id+1
      gap_ids = []
      if step_overrides is not None:
        step = step_overrides[chunk_id]
    
    gap_ids.append(step) 
    cycle_id = (cycle_id+1) % cycle_4095
  
  unique_patterns = [p for (p, f) in zip(patterns, first_occurences) if f]
  
  # Filter the unique patterns so that they don't consider gap probabilities
  # at the edges of the test chunks
  unique_patterns_no_edges = unique_patterns.copy()
  for i in range(len(unique_patterns_no_edges)):
    unique_patterns_no_edges[i] = unique_patterns_no_edges[i][np.logical_and(
        unique_patterns_no_edges[i] >= half_block_overlap,
        unique_patterns_no_edges[i] < chunk_size - half_block_overlap)]
  
  # Compute aggregate pattern probabilities for all test chunks
  num_unique_patterns = len(unique_patterns)
  num_test_files = gap_preds.shape[1]
  raw_pattern_mean_log_probs = np.zeros((num_unique_patterns, num_test_files))
  raw_pattern_mean_probs = np.zeros((num_unique_patterns, num_test_files))
  
  for i in range(num_unique_patterns):
    pat_probs = np.maximum(1e-8, gap_preds[unique_patterns_no_edges[i], :])
    raw_pattern_mean_log_probs[i] = np.log10(pat_probs).mean(0)
    raw_pattern_mean_probs[i] = pat_probs.mean(0)
    
  # Inspect which chunks have clear patterns
  most_likely_patterns = []
  second_most_likely_patterns = []
  oom_diffs_second_most_likely = []
  most_likely_log10_avg_probs = []
  most_likely_avg_probs = []
  avg_prob_ranks = []
  avg_prob_ratios_top_two = []
  for i in range(num_test_files):
    ordered_probs_patterns = np.argsort(-raw_pattern_mean_log_probs[:, i])
    two_most_likely_patterns = ordered_probs_patterns[:2]
    two_most_likely_mean_patterns = np.argsort(
        -raw_pattern_mean_probs[:, i])[:2]
    avg_prob_ranks.append(np.where(ordered_probs_patterns == (
        two_most_likely_mean_patterns[0]))[0][0]+1)
    avg_prob_ratios_top_two.append(
        raw_pattern_mean_probs[two_most_likely_mean_patterns[0], i]/
        raw_pattern_mean_probs[two_most_likely_mean_patterns[1], i])
    most_likely_patterns.append(two_most_likely_patterns[0])
    second_most_likely_patterns.append(two_most_likely_patterns[1])
    oom_diffs_second_most_likely.append(int(chunk_size/4096)*(
        raw_pattern_mean_log_probs[two_most_likely_patterns[0], i]-
        raw_pattern_mean_log_probs[two_most_likely_patterns[1], i]))
    most_likely_log10_avg_probs.append(raw_pattern_mean_log_probs[
          two_most_likely_patterns[0], i])
    most_likely_avg_probs.append(raw_pattern_mean_probs[
        two_most_likely_patterns[0], i])
    
  most_likely_prev_patterns = [(p - 37*4096 + chunk_size) % 4096 for p in (
      most_likely_patterns)]
  most_likely_next_patterns = [(p + 37*4096 - chunk_size) % 4096 for p in (
      most_likely_patterns)]
    
  pattern_summary = pd.DataFrame.from_dict({
      'test_file': test_file_names,
      'file_id': np.arange(num_test_files).tolist(),
      'most_likely_pattern': most_likely_patterns,
      'most_likely_log10_avg_prob': most_likely_log10_avg_probs,
      'most_likely_avg_prob': most_likely_avg_probs,
      'most_likely_prev_pattern': most_likely_prev_patterns,
      'most_likely_next_pattern': most_likely_next_patterns,
      'second_most_likely_pattern': second_most_likely_patterns,
      'oom_diffs_second_most_likely': oom_diffs_second_most_likely,
      'avg_prob_rank': avg_prob_ranks,
      'avg_prob_ratio_top_two': avg_prob_ratios_top_two,
      })
  pattern_summary = pattern_summary[[
      'file_id', 'test_file', 'most_likely_pattern',
      'most_likely_log10_avg_prob', 'most_likely_avg_prob',
      'second_most_likely_pattern',
      'oom_diffs_second_most_likely',
      'avg_prob_rank', 'avg_prob_ratio_top_two',
      'most_likely_prev_pattern', 'most_likely_next_pattern']]
  
  return pattern_summary, unique_patterns_no_edges, raw_pattern_mean_probs


## Loop over all possible initial steps and initial cycle_ids evaluate the
## resulting most likely pattern statistics
#pattern_summary = pattern_summary_initial_step(
#    initial_step=0, initial_cycle_id=0, cycle_4095=1280, num_chunks=2624)
#
if (not 'pattern_summary_all' in locals()):
  pattern_summary_all, unique_patterns_no_edges, raw_pattern_mean_probs = (
      pattern_summary_initial_step(
      gap_preds, initial_step=0, initial_cycle_id=1, cycle_4095=1280000,
      num_chunks=4096, step_overrides=np.arange(4097)))
  pattern_summary_all['chunk_std'] = chunk_std
  ttf_preds = pd.read_csv(ttf_file)
  
  if mode == 'test':
    pattern_summary_all['ttf_prediction'] = ttf_preds.time_to_failure.values
  else:
    pattern_summary_all['ttf'] = train_data.target.values[923:(923+2972)]
#
##pattern_summary_all = pattern_summary_initial_step(
##    initial_step=123, initial_cycle_id=123, cycle_4095=1280,
##    num_chunks=4096)
#
## Optimal for validation data
#pattern_summary_valid = pattern_summary_initial_step(
#    initial_step=0, initial_cycle_id=599, cycle_4095=1280,
#    num_chunks=gap_preds.shape[1])
#
##data_path = data_folder + 'most_likely_patterns' + pred_extension + '.csv'
##pattern_summary_all.to_csv(data_path, index=False)

# Consider the top K based on order probs and compute the offset modulo 4096
top_k_considered = 10
likely_gap_ids = np.zeros((4096, 5))
likely_gap_ids[:, 0] = np.arange(4096)
mlp = pattern_summary_all.most_likely_pattern.values
patt_diff = (4096 + np.expand_dims(mlp, 0) - np.expand_dims(mlp, 1) - 1552) % 4096
np.fill_diagonal(patt_diff, -1)
most_likely_files = np.zeros((num_files))
for i in range(num_files):
  most_likely_pattern = pattern_summary_all.most_likely_pattern.values[i]
  considered_next_files = sort_ids[i][:top_k_considered]
  most_likely_files[considered_next_files] += 1
  next_most_likely_patterns = pattern_summary_all.most_likely_pattern.values[
      considered_next_files]
  most_likely_gaps = (next_most_likely_patterns - most_likely_pattern - 1552 +(
      2*4096)) % 4096
  likely_gap_ids[most_likely_gaps, 1] = likely_gap_ids[most_likely_gaps, 1] + 1
  likely_gap_ids[most_likely_gaps, 2] = likely_gap_ids[most_likely_gaps, 2] + (
      order_preds_mean[i, sort_ids[i][:top_k_considered]])
likely_gap_ids[:, 2] /= likely_gap_ids[:, 1]
  
for i in range(4096):
  gap_match_ids = (patt_diff==i)
  likely_gap_ids[i, 3] = gap_match_ids.sum()#/(num_files-1)
  likely_gap_ids[i, 4] = order_preds_mean[gap_match_ids].mean()
  
likely_gap_ids_sorted = likely_gap_ids[(-likely_gap_ids[:, 1]).argsort()]  