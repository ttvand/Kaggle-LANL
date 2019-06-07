import numpy as np
import pandas as pd


data_folder = '/media/tom/cbd_drive/Kaggle/LANL/Data/'
submission_folder = '/home/tom/Kaggle/LANL/Submissions/'
submissions = [
    '19-06-03-14-17 - FINAL_SUBMISSION_1 - last_six lgbm nn blend - median_test_cycle_length 12.csv',
    '19-06-03-14-18 - FINAL_SUBMISSION_2 - all_prev lgbm nn blend - median_test_cycle_length 12.csv',
    ]

target_mean = 6.6

num_submissions = len(submissions)
model_preds = np.zeros((2624, num_submissions))
for i in range(num_submissions):
  print('Loading test predictions {} of {}'.format(i+1, num_submissions))
  submission_preds = pd.read_csv(
      submission_folder + submissions[i]).time_to_failure.values
  preds_test = submission_preds * target_mean/submission_preds.mean()

  # Write the output pandas data frame
  submission = pd.read_csv(data_folder + 'sample_submission.csv')
  submission.time_to_failure = preds_test
  submission_path = submission_folder + submissions[i][:-4] + '_RESCALED_' + (
      str(target_mean)) + '.csv'
  submission.to_csv(submission_path, index=False)