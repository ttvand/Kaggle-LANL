import gc
import numpy as np
import os
import pandas as pd
import pickle
import utils

from collections import OrderedDict
from joblib import Parallel, delayed
from scipy.signal import savgol_filter
from scipy.stats import boxcox
from scipy.stats import kurtosis
from tqdm import tqdm


# Set the path to the data folder
data_folder = '/media/tom/cbd_drive/Kaggle/LANL/Data/'
#data_folder = '/home/tom/Kaggle/LANL/Data/'


class FeatureGenerator(object):
  def __init__(self, dtype, target_quantile, n_jobs, chunk_size=150000,
               chunk_overlap_size=37500):
    gc.collect()
    self.dtype = dtype
    self.target_quantile = target_quantile
    self.n_jobs = n_jobs
    self.chunk_size = chunk_size
    self.chunk_overlap_size = chunk_overlap_size
    self.test_files = []
    if self.dtype == 'train':
      self.train_data = pd.read_csv(
          data_folder + 'train.csv', dtype={'acoustic_data': np.float64,
                                            'time_to_failure': np.float64})
      self.store_train_start_rows()
      self.num_chunks = len(self.train_start_rows)
    else:
      submission = pd.read_csv(data_folder + 'sample_submission.csv')
      for seg_id in submission.seg_id.values:
        self.test_files.append(
            (seg_id, data_folder + 'test/' + seg_id + '.csv'))
      self.num_chunks = len(submission)

  def store_train_start_rows(self):
    ttf = self.train_data.time_to_failure.values
    eq_ids = np.append(np.where(np.diff(ttf) > 0)[0]+1, ttf.size)
    self.train_start_rows = []
    self.earthquake_ids = []
    self.no_overlap_chunks = []
    start_id = 0
    eq_id = 0
    while start_id < (ttf.size - self.chunk_size - 1):
      if start_id + self.chunk_size == eq_ids[eq_id]:
        self.earthquake_ids.append(eq_id)
        self.train_start_rows.append(start_id)
        self.no_overlap_chunks.append(True)
        start_id = int(eq_ids[eq_id])
        eq_id = eq_id+1
      elif start_id + self.chunk_size > eq_ids[eq_id]:
        self.earthquake_ids.append(eq_id)
        start_id = int(eq_ids[eq_id])
        self.train_start_rows.append(start_id - self.chunk_size)
        self.no_overlap_chunks.append(True)
        eq_id = eq_id+1
      else:
        considered_start_rows = start_id + np.arange(
            0, self.chunk_size, self.chunk_overlap_size)
        valid_start_rows = considered_start_rows[
            considered_start_rows <= eq_ids[eq_id] - self.chunk_size]
        valid_start_rows = valid_start_rows.tolist()
        self.train_start_rows.extend(valid_start_rows)
        self.earthquake_ids.extend([eq_id]*len(valid_start_rows))
        self.no_overlap_chunks.append(True)
        self.no_overlap_chunks.extend([False]*(len(valid_start_rows)-1))
        start_id = start_id + self.chunk_size
    
    self.earthquake_ids.append(eq_id)
    self.train_start_rows.append(ttf.size - self.chunk_size)
    self.no_overlap_chunks.append(True)

  def read_chunks(self):
    if self.dtype == 'train':
      for counter, start_row in enumerate(self.train_start_rows):
        df = self.train_data.iloc[start_row:(
            start_row + self.chunk_size)]
        x = df.acoustic_data.values
        y = df.time_to_failure.values[-1]
        y_original = y
        if self.target_quantile:
          eq_id = self.earthquake_ids[counter]
          if eq_id == 0:
            y = np.nan
          else:
            cycle_length = self.train_data.iloc[self.train_start_rows[np.where(
                np.array(self.earthquake_ids) == eq_id)[0][
          0]+150000]].time_to_failure
            y = (cycle_length-df.time_to_failure.values.min())/cycle_length
        seg_id = 'train_' + str(counter) + '_eq_' + str(
            self.earthquake_ids[counter])
        no_overlap_chunk = self.no_overlap_chunks[counter]
        # Set all tr. weights to 1
        yield x, y, y_original, seg_id, 1, no_overlap_chunk, start_row
    else:
      for seg_id, f in self.test_files:
        df = pd.read_csv(f, dtype={'acoustic_data': np.float64})
        x = df.acoustic_data.values
        yield x, -999, -999, seg_id, -999, True, -999

  def get_features(self, x, y, y_original, seg_id, weight, no_overlap_chunk,
                   start_row):
    features = OrderedDict()
    features['target'] = y
    features['notrain_target_original'] = y_original
    features['notrain_seg_id'] = seg_id
    eq_id = -999
    if seg_id[:5] == 'train':
      eq_id = int(seg_id.split('_')[3])
    features['notrain_eq_id'] = eq_id
    features['notrain_train_weight'] = weight
    features['notrain_no_overlap_chunk'] = no_overlap_chunk
    features['notrain_start_row'] = start_row
    custom_features = self.custom_features(x)
    features.update(custom_features)

    return features
    
  def custom_features(self, x, sub_chunk_size=1500):
    # Create custom features here
    # 75,000 should be divisible by sub_chunk_size
    features = OrderedDict()
    
    chunk_resized = x.reshape([1, -1])
    sub_chunk = x.reshape([-1, sub_chunk_size])
    
    features.update(self.general_features(chunk_resized))
    features.update(self.freq_features(chunk_resized, sub_chunk_size))
    features.update(self.general_features(sub_chunk, 'sub'))
    features.update(self.freq_features(sub_chunk, sub_chunk_size, 'sub'))
    
    return features
  
  def freq_features(self, x, sub_chunk_size, feature_type='chunk'):
    # Apply FFT and compute the amplitude of the frequencies
    chunk_fft_amp = np.abs(np.fft.fft(x, axis=1)[:, :x.shape[1] // 2])
    chunk_fft_amp = chunk_fft_amp.reshape([chunk_fft_amp.shape[0],
                                           sub_chunk_size // 2, -1]).mean(2)

    # Aggregate the amplitude frequencies
    agg_ids = np.cumsum(2**(np.arange(np.floor(np.log2(sub_chunk_size))-1)),
                        dtype=np.int32)
    num_agg_ids = agg_ids.size
    num_steps = chunk_fft_amp.shape[0]
    freq_features = np.zeros((num_steps, num_agg_ids+1))

    start_id = 0
    for i in range(num_agg_ids):
      end_id = agg_ids[i]
      freq_features[:, i] = np.mean(chunk_fft_amp[:, start_id:end_id], 1)
      start_id = end_id
    freq_features[:, -1] = np.mean(chunk_fft_amp[:, start_id:], 1)
    
    # Name the features
    keys = []
    for f in range(num_agg_ids+1):
      for s in range(num_steps):
        if num_steps == 1:
          keys.append(feature_type + '_fft_' + str(f))
        else:
          keys.append(feature_type + '_fft_' + str(f) + '_sub_step_' + str(s))
    
    return utils.ordered_dict(keys,
                              freq_features.transpose().flatten().tolist())
  
  def general_features(self, x, feature_type='chunk',
                       q_vals=[0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99,
                               0.999], q_abs_vals=[0.8, 0.9, 0.99, 0.999, 1],
                               autocorr_lags = [2, 3, 5, 8, 20, 100],
                               peak_thresholds=[5, 10, 20, 100]):
    vals = []
    names = []
    x_min4 = x-4
    num_subchunks = x.shape[0]
    vals.extend(np.mean(x, axis=1).tolist()); names.append('mean')
    vals.extend(np.std(x, axis=1).tolist()); names.append('std')
    vals.extend(np.argmax(np.abs(x), axis=1).tolist()); names.append(
        'notrain_abs_argmax')
    vals.extend(np.mean(np.abs(x_min4) > 10, axis=1).tolist()); names.append(
        'frac_greater_10')
    vals.extend(np.mean(np.abs(x_min4) > 100, axis=1).tolist()); names.append(
        'frac_greater_100')
    vals.extend(np.mean(np.abs(x_min4) > 1000, axis=1).tolist()); names.append(
        'frac_greater_1000')
    quantiles = np.quantile(x_min4, q_vals, axis=1)
    abs_quantiles = np.quantile(np.abs(x_min4), q_abs_vals, axis=1)
    for i in range(len(q_vals)):
      vals.extend(quantiles[i].tolist()); names.append('q_' + str(q_vals[i]))
    for i in range(len(q_abs_vals)):
      vals.extend(abs_quantiles[i].tolist())
      names.append('q_abs_' + str(q_abs_vals[i]))
    
    # Compute the number of peaks for different thresholds
    num_peak_thresholds = len(peak_thresholds)
    window_size = 10
    peak_counts = np.zeros((num_peak_thresholds, num_subchunks))
    for i in range(num_subchunks):
      x_abs_smoothed = savgol_filter(x[i]-np.median(x[i]), 51, 3)
      for peak_id, threshold in enumerate(peak_thresholds):
        threshold_exceed_ids = np.where(np.logical_and(
            x_abs_smoothed[:-window_size] < threshold,
            x_abs_smoothed[window_size:] > threshold*1.1))[0]
        peak_count = 0
        if threshold_exceed_ids.size:
          peak_count = 1
          last_exceed_id = threshold_exceed_ids[0]
          for exceed_id in threshold_exceed_ids[1:]:
            if exceed_id > (last_exceed_id + 10):
              peak_count += 1
              last_exceed_id = exceed_id
        peak_counts[peak_id, i] = peak_count
    for i in range(num_peak_thresholds):
      vals.extend(peak_counts[i].tolist())
      names.append('peak_counts_' + str(peak_thresholds[i]))
    
    # Compute autocorrelation statistics
#    x = np.random.normal(size=(100, 1500)); autocorr_lags=[2, 3, 5, 8, 20]; num_subchunks=100
    num_autocorr = len(autocorr_lags)
    autocorrs = np.zeros((num_autocorr, num_subchunks))
    max_autocorr = np.max(autocorr_lags)
    for i in range(num_subchunks):
      x_autocorr = np.zeros((1+num_autocorr, x.shape[1]-max_autocorr))
      x_autocorr[0] = x[i, :-max_autocorr]
      for lag_id, lag in enumerate(autocorr_lags):
        x_autocorr[1+lag_id] = x[i, lag:(x.shape[1]-max_autocorr+lag)]
      autocorrs[:, i] = np.corrcoef(x_autocorr)[1:, 1]
    for i in range(num_autocorr):
      vals.extend(autocorrs[i].tolist())
      names.append('autocorr_' + str(autocorr_lags[i]))
    
    # Compute the kurtosis of the distribution
    vals.extend(kurtosis(x, axis=1).tolist()); names.append('kurtosis')
    
    # Name the features
    keys = []
    num_steps = x.shape[0]
    for n in names:
      for s in range(num_steps):
        if num_steps == 1:
          keys.append(feature_type + '_' + n)
        else:
          keys.append(feature_type + '_' + n + '_sub_step_' + str(s))
    
    return utils.ordered_dict(keys, vals)

  def generate(self):
    feature_list = []
    res = Parallel(n_jobs=self.n_jobs, backend='threading')(
        delayed(self.get_features)(x, y, yo, s, w, o, r) for (
            x, y, yo, s, w, o, r) in tqdm(self.read_chunks(),
                                         total=self.num_chunks))
    if self.dtype == 'train':
      self.train_data = None
      gc.collect()
    for r in res:
      feature_list.append(r)
    if self.dtype == 'train':
      del self.train_data
    return pd.DataFrame(feature_list)
      
      
def update_od(od1, od2):
  od1 = OrderedDict(list(od1.items()) + list(od2.items()))

#fg = FeatureGenerator(dtype='test', n_jobs=None)
#fg.custom_features(np.random.randint(-100, 100, size=(150000)))

def get_preprocessed(source, remove_overlap_chunks, target_quantile=False,
                     remove_incomplete_eqs=True, scale=False,
                     train_last_six_complete=False):
  scaled_ext = '_scaled' if scale else ''
  keep_incomplete_ext = '_keep_incomplete' if not remove_incomplete_eqs else ''
  quantile_ext = '_target_quantile' if target_quantile else ''
  data_path = data_folder + source + '_features' + scaled_ext + (
      keep_incomplete_ext + quantile_ext + '.csv')
  if os.path.exists(data_path):
    data = pd.read_csv(data_path)
  else:
    if scale:
      # Read the original train and test csvs and scale the training features
      # Calling the unscaled version makes sure that the unscaled features
      # are present
      _, _, _ = get_preprocessed(
          'test', remove_overlap_chunks=False, scale=False,
          remove_incomplete_eqs=remove_incomplete_eqs,
          train_last_six_complete=False, target_quantile=True, )
      train_features_unscaled, _, _ = get_preprocessed(
          'train', remove_overlap_chunks=False, scale=False,
          remove_incomplete_eqs=remove_incomplete_eqs,
          train_last_six_complete=False, target_quantile=True)
      scale_cols = train_features_unscaled.columns
      train_unscaled = pd.read_csv(
          data_folder + 'train_features_keep_incomplete_target_quantile.csv')
      test_unscaled = pd.read_csv(
          data_folder + 'test_features_keep_incomplete_target_quantile.csv')
      train_test = pd.concat([train_unscaled, test_unscaled])
      
      # Apply the Box-Cox transform to the scale cols
      # It was verified that the transformation is deterministic so the 
      # transformation is the same for train and test data!
      def normalize_fun(x):
        x_shifted = x - x.min() + 1
        xt, _ = boxcox(x_shifted)
        if xt.mean() > 1e10:
          xt = np.log(np.quantile(x_shifted, 0.01) + x_shifted)
        return np.clip((xt - xt.mean()) / xt.std(), -10, 10)
      train_test[scale_cols] = train_test[scale_cols].apply(normalize_fun)
      
      # Store the relevant part of train_test
      cutoff = train_unscaled.shape[0]
      data = train_test[:cutoff] if source == 'train' else train_test[cutoff:]
    else:
      fg = FeatureGenerator(dtype=source, target_quantile=target_quantile,
                            n_jobs=-2)
      data = fg.generate()
      del fg
    data.to_csv(data_path, index=False)
  
  # Drop incomplete earthquake data to avoid biasing the data to initial and
  # final parts of the earthquake cycle
  if remove_incomplete_eqs:
    eq_ids = data.notrain_eq_id.values
    incomplete_cycle = np.logical_or(eq_ids == 0, eq_ids == 16)
    data = data[~incomplete_cycle]
    
  if train_last_six_complete:
    eq_ids = data.notrain_eq_id.values
    keep_cycle_ids = eq_ids >= 10
    data = data[keep_cycle_ids]
    
  if remove_overlap_chunks:
    no_overlap_chunks = data.notrain_no_overlap_chunk.values
    data = data[no_overlap_chunks]
  
  cols = data.columns.tolist()
  other_cols = [c for c in cols if 'notrain_' in c]
  train_cols = [c for c in cols if not 'notrain_' in c and not (c == 'target')]
  
  other_data = data[other_cols]  
  other_data.columns = [c[8:] for c in other_data.columns] 
  
  target = data.target.values
  
  return data[train_cols], other_data, target


def train_val_split(
    remove_overlap_chunks, seed=14, num_folds=5, num_splits=10,
    remove_incomplete_eqs=True, ordered=False, train_all_previous=False,
    train_val_file_base=data_folder + 'train_val_split',
    target_quantile=False, train_last_six_complete=False):
  overlap_chunks_ext = '_remove_overlap' if remove_overlap_chunks else ''
  ordered_ext = '_ordered' if ordered else ''
  prev_ext = '_all_prev' if train_all_previous else ''
  quant_ext = '_target_quantile' if target_quantile else ''
  last_six_ext = '_last_six' if train_last_six_complete else ''
  train_val_file = train_val_file_base + overlap_chunks_ext + ordered_ext + (
      prev_ext + quant_ext + last_six_ext + '.pickle')
  if os.path.exists(train_val_file):
    with open(train_val_file, 'rb') as f:
      train_val_ids = pickle.load(f)
  else:
    num_splits = 1 if ordered else num_splits
    preprocessed = get_preprocessed(
        'train', remove_incomplete_eqs=remove_incomplete_eqs,
        remove_overlap_chunks=remove_overlap_chunks,
        target_quantile=target_quantile,
        train_last_six_complete=train_last_six_complete)
    train_other_cols = preprocessed[1]
    targets = train_other_cols.target_original.values
    eq_ids = train_other_cols.eq_id.values
    unique_eq_ids = np.unique(eq_ids)
    num_eqs = unique_eq_ids.size
    train_val_ids = []
    for i in range(num_splits):
      train_val_ids_split = []
      
      # Set the random seed in order to make eq ids consistent for both 
      # settings of remove_incomplete_eqs
      np.random.seed(seed + i)
      
      # Shuffle the earthquake ids for validation and early stopping selection
      perm = unique_eq_ids if ordered else np.random.permutation(unique_eq_ids)
      
      val_end = 0
      import pdb; pdb.set_trace()
      for j in range(num_folds):
        val_start = val_end
        val_end = int(np.around(num_eqs*(j+1)/num_folds))
        
        validation_bools = np.isin(eq_ids, perm[val_start:val_end])
        if ordered:
          train_bools = eq_ids <= perm[val_end-1] if train_all_previous else (
              validation_bools)
          train_bools = np.logical_and(train_bools, ~np.isnan(preprocessed[2]))
          train_eq_range = np.array([eq_ids[train_bools].min(),
                                     eq_ids[train_bools].max()])
          val_eqs = perm[val_end:]
          unique_val_eqs = np.unique(val_eqs).tolist()
          val_means_eq = np.array([])
          val_means_duration = np.array([])
          for eq in unique_val_eqs:
            val_means_eq = np.append(
                val_means_eq, np.mean(targets[eq == eq_ids]))
            val_means_duration = np.append(
                val_means_duration, targets[eq == eq_ids].ptp())
          val_means = np.sum(val_means_eq*val_means_duration)/(np.sum(
              val_means_duration) + 1e-9) # Avoid 0/0
          validation_bools = eq_ids > perm[val_end-1]
          train_ids = np.random.permutation(np.where(train_bools)[0])
          val_ids = np.random.permutation(np.where(validation_bools)[0])
          cyle_med_reps = []
          targets_validation = targets[val_ids]
          for (i, eq) in enumerate(val_eqs):
            cycle_length = targets[eq == eq_ids].max()
            reps = (eq == eq_ids).sum()
            cyle_med_reps.append(np.repeat(cycle_length, reps))
          if cyle_med_reps:
            median_cycle_duration = np.median(np.hstack(cyle_med_reps))
          else:
            median_cycle_duration = -999
          train_val_ids_split.append(
              (train_ids, val_ids, train_eq_range, val_eqs, val_means_eq,
               val_means_duration, val_means, median_cycle_duration,
               targets_validation))
        else:
          val_ids = np.where(validation_bools)[0]
          train_ids = np.random.permutation(
              np.where(np.logical_not(validation_bools))[0])
          
          es_eq_id = perm[val_end] if val_start == 0 else perm[val_start-1]
          es_bools = eq_ids == es_eq_id
          es_ids = np.where(es_bools)[0]
          train_noes_ids = np.where(np.logical_and(
              np.logical_not(validation_bools), np.logical_not(es_bools)))[0]
          
          train_val_ids_split.append((train_ids, val_ids, train_noes_ids,
                                      es_ids))
      
      train_val_ids.append(train_val_ids_split)
    train_val_ids = train_val_ids[0] if ordered else train_val_ids
      
    # Save the train validation ids
    with open(train_val_file, 'wb') as f:
      pickle.dump(train_val_ids, f, protocol=pickle.HIGHEST_PROTOCOL)
    
  return(train_val_ids)
  
  
def train_val_split_gaps(
    seed=14, num_folds=5, num_splits=10, remove_incomplete_eqs=True,
    train_val_file=data_folder + 'train_val_split_gaps.pickle',
    target_quantile=False):
  if os.path.exists(train_val_file):
    with open(train_val_file, 'rb') as f:
      train_val_ids = pickle.load(f)
  else:
    other_train_features = get_preprocessed(
        'train', remove_incomplete_eqs=remove_incomplete_eqs,
        remove_overlap_chunks=True, target_quantile=target_quantile)[1]
    start_rows = other_train_features.start_row.values
    chunk_train_val_split = train_val_split(
        remove_overlap_chunks=True,
        remove_incomplete_eqs=remove_incomplete_eqs)
    
    train_val_ids = []
    for i in range(num_splits):
      train_val_ids_split = []
      
      for j in range(num_folds):
        fold_ids = chunk_train_val_split[i][j]
        train_val_ids_split.append((get_cont_ranges(fold_ids[0], start_rows),
                                    get_cont_ranges(fold_ids[1], start_rows)))
        
      train_val_ids.append(train_val_ids_split)
      
    # Save the train validation ids
    with open(train_val_file, 'wb') as f:
      pickle.dump(train_val_ids, f, protocol=pickle.HIGHEST_PROTOCOL)
    
  return(train_val_ids)
  
  
def get_cont_ranges(ids, vals):
  sorted_ids = np.sort(ids)
  jump_ids = np.where(np.diff(sorted_ids) > 1)[0]
  start_ids = np.array([sorted_ids[0]] + sorted_ids[jump_ids+1].tolist())
  end_ids = np.array(sorted_ids[jump_ids].tolist() + [sorted_ids[-1]])
  return (vals[start_ids], vals[end_ids]+150000)


## Only run the logic below once to save time
#train_val_split(remove_overlap_chunks=False, ordered=False)
#train_val_split(remove_overlap_chunks=True, ordered=False)
#train_val_split(remove_overlap_chunks=False, ordered=True)
#train_val_split(remove_overlap_chunks=True, ordered=True)
#train_val_split_gaps()
  