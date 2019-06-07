import numpy as np
import pandas as pd

data_folder = '/media/tom/cbd_drive/Kaggle/LANL/Data/'
gap_data = pd.read_csv(data_folder + 'train.csv')

ttf_diff = np.diff(gap_data.time_to_failure.values)
gap_data['is_gap'] = np.concatenate(
    ([False], np.logical_or(ttf_diff < -1e-4, ttf_diff > 0)))
gap_data = gap_data.drop("time_to_failure", axis=1)

data_path = data_folder + 'gap_data.csv'
gap_data.to_csv(data_path, index=False)
