import numpy as np
import pandas as pd

data_folder = '/media/tom/cbd_drive/Kaggle/LANL/Data/'
mode = ['validation', 'test'][1]
pred_extension = '_test' if mode == 'test' else '_valid'
base_pred_path = '../Models/Best models/gap_model_aligned_predictions'
pred_path = base_pred_path + pred_extension + '.csv'

# Load the gap predictions for all test chunks
if (not 'gap_preds_test' in locals()):
  gap_preds_test = pd.read_csv(pred_path)
  test_file_names = list(gap_preds_test.columns.values)
  gap_preds_test = gap_preds_test.values

# List all possible gap patterns for the test chunks given the initial step
def pattern_summary_initial_step(initial_step=0, initial_cycle_id=0,
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
  num_test_files = gap_preds_test.shape[1]
  raw_pattern_mean_log_probs = np.zeros((num_unique_patterns, num_test_files))
  raw_pattern_mean_probs = np.zeros((num_unique_patterns, num_test_files))
  
  for i in range(num_unique_patterns):
    pat_probs = gap_preds_test[unique_patterns_no_edges[i], :]
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
      'second_most_likely_pattern', 'oom_diffs_second_most_likely',
      'avg_prob_rank', 'avg_prob_ratio_top_two',
      'most_likely_prev_pattern', 'most_likely_next_pattern']]
  
  return pattern_summary


# Loop over all possible initial steps and initial cycle_ids evaluate the
# resulting most likely pattern statistics
pattern_summary = pattern_summary_initial_step(
    initial_step=0, initial_cycle_id=0, cycle_4095=1280, num_chunks=2624)

pattern_summary_all = pattern_summary_initial_step(
    initial_step=0, initial_cycle_id=1, cycle_4095=1280000, num_chunks=4096,
    step_overrides=np.arange(4097))

#pattern_summary_all = pattern_summary_initial_step(
#    initial_step=123, initial_cycle_id=123, cycle_4095=1280,
#    num_chunks=4096)

# Optimal for validation data
pattern_summary_valid = pattern_summary_initial_step(
    initial_step=0, initial_cycle_id=599, cycle_4095=1280,
    num_chunks=gap_preds_test.shape[1])

#data_path = data_folder + 'most_likely_patterns' + pred_extension + '.csv'
#pattern_summary_all.to_csv(data_path, index=False)
