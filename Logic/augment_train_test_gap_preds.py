import gc
import numpy as np
import pandas as pd

data_folder = '/media/tom/cbd_drive/Kaggle/LANL/Data/'
signal_files = {'train': 'train.csv', 'test': 'test_combined.csv'}
for source in ['test', 'train']:
  signal_data = pd.read_csv(data_folder + signal_files[source])
  
  # Normalize the signal data
  vals = signal_data.acoustic_data.values
  signal_data.acoustic_data = np.sign(vals-5) * np.log10(1+np.abs(vals-5))
  
  gap_file = 'gap_model_aligned_predictions_' + source
  gap_data = pd.read_csv(data_folder + gap_file + '.csv')
  
  # Insert the gap predictions into the signal data
  gap_preds = np.clip(np.reshape(gap_data.values, [-1]), 1e-10, 1)
  signal_data.insert(1, 'gap_log_prediction', 4+np.log10(gap_preds))
  
  # Store the augmented data
  del gap_data; gc.collect()
  for col in signal_data.columns:
    signal_data[col] = signal_data[col].astype(np.float32)
  data_path = data_folder + source + '_normalized_gap_augment.csv'
  signal_data.to_csv(data_path, index=False)

  # Try to free up memory
  del signal_data; gc.collect()