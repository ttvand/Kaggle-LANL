import numpy as np
import pandas as pd

data_folder = '/media/tom/cbd_drive/Kaggle/LANL/Data/'

if not 'test_combined' in locals():
  test_combined = pd.read_csv(
      data_folder + 'test_combined.csv').values.reshape(-1, 150000)

extreme_length = 10000
first_id = 0

def get_max_shared_seq_len(first_id, sec_id):
  valid_ids = np.where(
      start_test[sec_id] == end_test[first_id, extreme_length-1])[0]
  longest_match = 0
  while valid_ids.size:
    longest_match += 1
    valid_ids = valid_ids[np.logical_and(
        valid_ids >= longest_match,
        end_test[first_id, extreme_length-longest_match-1] == start_test[
            sec_id, valid_ids-longest_match])]
  return longest_match

#longest_match_sequence.max()
#longest_match_sequence.argmax()
#get_max_shared_seq_len(1418, 1232) # Longest sequence

num_test_files = test_combined.shape[0]
start_test = test_combined[:, :extreme_length]
end_test = test_combined[:, -extreme_length:]
longest_match_sequence = np.zeros((num_test_files, num_test_files))
for first_id in range(num_test_files):
  print(first_id)
  for sec_id in range(num_test_files):
    longest_match_sequence[first_id, sec_id] = get_max_shared_seq_len(
        first_id, sec_id)
  
