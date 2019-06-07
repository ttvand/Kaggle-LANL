import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


data_folder = '/media/tom/cbd_drive/Kaggle/LANL/Data/'
submission_folder = '/home/tom/Kaggle/LANL/Submissions/'
submissions = [
    '19-06-03-14-17 - FINAL_SUBMISSION_1 - last_six lgbm nn blend - median_test_cycle_length 12.csv',
    '19-06-03-14-18 - FINAL_SUBMISSION_2 - all_prev lgbm nn blend - median_test_cycle_length 12.csv',
    '19-06-02-22-20 - Average LightGBM NN median_test_cycle 11.0.csv',
    
    
    
    '19-06-03-11-39 - lgbm_last_six_seed_1_of_10_12.0.csv',
    '19-06-03-11-39 - lgbm_last_six_seed_2_of_10_12.0.csv',
    '19-06-03-11-40 - lgbm_last_six_seed_3_of_10_12.0.csv',
    '19-06-03-11-40 - lgbm_last_six_seed_4_of_10_12.0.csv',
    '19-06-03-11-40 - lgbm_last_six_seed_5_of_10_12.0.csv',
    '19-06-03-11-41 - lgbm_last_six_seed_6_of_10_12.0.csv',
    '19-06-03-11-41 - lgbm_last_six_seed_7_of_10_12.0.csv',
    '19-06-03-11-42 - lgbm_last_six_seed_8_of_10_12.0.csv',
    '19-06-03-11-42 - lgbm_last_six_seed_9_of_10_12.0.csv',
    '19-06-03-11-42 - lgbm_last_six_seed_10_of_10_12.0.csv',
    
    '19-06-03-11-54 - nn_last_six_seed_1_of_10_12.0.csv',
    '19-06-03-11-55 - nn_last_six_seed_2_of_10_12.0.csv',
    '19-06-03-11-57 - nn_last_six_seed_3_of_10_12.0.csv',
    '19-06-03-11-58 - nn_last_six_seed_4_of_10_12.0.csv',
    '19-06-03-11-59 - nn_last_six_seed_5_of_10_12.0.csv',
    '19-06-03-12-00 - nn_last_six_seed_6_of_10_12.0.csv',
    '19-06-03-12-02 - nn_last_six_seed_7_of_10_12.0.csv',
    '19-06-03-12-03 - nn_last_six_seed_8_of_10_12.0.csv',
    '19-06-03-12-04 - nn_last_six_seed_9_of_10_12.0.csv',
    '19-06-03-12-06 - nn_last_six_seed_10_of_10_12.0.csv',
    
    
    
    '19-06-03-11-43 - lgbm_all_prev_seed_1_of_10_12.0.csv',
    '19-06-03-11-44 - lgbm_all_prev_seed_2_of_10_12.0.csv',
    '19-06-03-11-45 - lgbm_all_prev_seed_3_of_10_12.0.csv',
    '19-06-03-11-46 - lgbm_all_prev_seed_4_of_10_12.0.csv',
    '19-06-03-11-48 - lgbm_all_prev_seed_5_of_10_12.0.csv',
    '19-06-03-11-49 - lgbm_all_prev_seed_6_of_10_12.0.csv',
    '19-06-03-11-50 - lgbm_all_prev_seed_7_of_10_12.0.csv',
    '19-06-03-11-51 - lgbm_all_prev_seed_8_of_10_12.0.csv',
    '19-06-03-11-52 - lgbm_all_prev_seed_9_of_10_12.0.csv',
    '19-06-03-11-53 - lgbm_all_prev_seed_10_of_10_12.0.csv',
    
    '19-06-03-12-13 - nn_all_prev_seed_1_of_10_12.0.csv',
    '19-06-03-12-21 - nn_all_prev_seed_2_of_10_12.0.csv',
    '19-06-03-12-28 - nn_all_prev_seed_3_of_10_12.0.csv',
    '19-06-03-12-36 - nn_all_prev_seed_4_of_10_12.0.csv',
    '19-06-03-12-43 - nn_all_prev_seed_5_of_10_12.0.csv',
    '19-06-03-12-50 - nn_all_prev_seed_6_of_10_12.0.csv',
    '19-06-03-12-58 - nn_all_prev_seed_7_of_10_12.0.csv',
    '19-06-03-13-05 - nn_all_prev_seed_8_of_10_12.0.csv',
    '19-06-03-13-13 - nn_all_prev_seed_9_of_10_12.0.csv',
    '19-06-03-13-20 - nn_all_prev_seed_10_of_10_12.0.csv',
    ]


num_submissions = len(submissions)
model_preds = np.zeros((2624, num_submissions))
for i in range(num_submissions):
  print('Loading test predictions {} of {}'.format(i+1, num_submissions))
  submission = pd.read_csv(submission_folder + submissions[i])
  model_preds[:, i] = submission.time_to_failure.values
preds_test = np.mean(np.maximum(0, model_preds), 1)
corr_matrix = np.corrcoef(np.transpose(model_preds))
plt.hist(preds_test, bins=30)
plt.show()

# Write the output pandas data frame
submission = pd.read_csv(data_folder + 'sample_submission.csv')
submission.time_to_failure = preds_test
the_date = datetime.datetime.now().strftime('%y-%m-%d-%H-%M')
submission_path = submission_folder + the_date + '.csv'
import pdb; pdb.set_trace()
submission.to_csv(submission_path, index=False)