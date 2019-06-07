import pandas as pd

data_folder = '/media/tom/cbd_drive/Kaggle/LANL/Data/'

submission = pd.read_csv(data_folder + 'sample_submission.csv')
test_files = []
for seg_id in submission.seg_id.values:
  test_files.append(data_folder + 'test/' + seg_id + '.csv')
num_chunks = len(submission)

test_dfs = []
for test_file in test_files:
  test_dfs.append(pd.read_csv(test_file))
  
test_combined = pd.concat(test_dfs)

data_path = data_folder + 'test_combined.csv'
test_combined.to_csv(data_path, index=False)
