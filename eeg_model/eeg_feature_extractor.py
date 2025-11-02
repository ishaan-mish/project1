#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import scipy
import scipy.signal
import scipy.stats
import scipy.linalg
import pandas as pd # Added import for pandas for matrix_from_csv_file
from datetime import datetime # Added import for datetime logging

# --------------------- Data Reading and Slicing ---------------------

def matrix_from_csv_file(file_path):
    # print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: Reading CSV file: {file_path}")
    try:
        # Using pandas for more robust CSV reading as np.genfromtxt can struggle with partial/malformed files
        df = pd.read_csv(file_path)
        # Convert DataFrame values to numpy array. Assuming headers were read by pandas
        full_matrix = df.values 
        # print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: CSV data matrix shape: {full_matrix.shape}")
        
        # Ensure matrix is not empty and has expected columns (Timestamp, AF7, TP9)
        if full_matrix.shape[0] < 1 or full_matrix.shape[1] < 3: 
            # print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: WARNING: Insufficient data rows or columns in CSV: {full_matrix.shape}. Returning empty array.")
            return np.array([])
        return full_matrix
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: ERROR reading CSV file {file_path}: {e}")
        return np.array([]) # Return empty array on error


def get_time_slice(full_matrix, start=0., period=1.):
    # print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: get_time_slice called with start={start}, period={period}")
    if full_matrix.size == 0:
        # print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: get_time_slice: full_matrix is empty. Raising IndexError.")
        raise IndexError("Full matrix is empty")
    
    # Ensure timestamp column (index 0) is treated as numeric
    try:
        timestamps = full_matrix[:, 0].astype(float) 
    except ValueError as ve:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: get_time_slice: Error converting timestamps to float: {ve}. Data might be malformed.")
        raise ValueError(f"Timestamp conversion error: {ve}")

    # Use first timestamp in the matrix as reference point for relative start
    rstart = timestamps[0] + start
    
    # Find indices that correspond to the time slice
    indices_le_rstart = np.where(timestamps <= rstart)[0]
    index_0 = indices_le_rstart[-1] if indices_le_rstart.size > 0 else 0

    indices_le_rstart_plus_period = np.where(timestamps <= rstart + period)[0]
    index_1 = indices_le_rstart_plus_period[-1] if indices_le_rstart_plus_period.size > 0 else len(timestamps) - 1

    # Adjust index_1 to ensure it's not beyond the last available sample
    index_1 = min(index_1, len(timestamps) - 1)

    # Ensure index_1 is at least index_0 for a valid slice
    if index_1 < index_0:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: get_time_slice: index_1 ({index_1}) < index_0 ({index_0}). No valid slice. Raising IndexError.")
        raise IndexError("No valid time slice found.")

    slice_data = full_matrix[index_0:index_1 + 1, :] # Include index_1 in slice
    
    if slice_data.shape[0] == 0:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: get_time_slice: Slice data is empty after indexing. Indices: {index_0}:{index_1+1}. Raising IndexError.")
        raise IndexError("Slice data is empty.")

    duration = timestamps[index_1] - timestamps[index_0]
    # print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: get_time_slice: Slice from index {index_0} to {index_1}, duration {duration:.2f}s, shape {slice_data.shape}")
    return slice_data, duration

# --------------------- Feature Extraction Functions ---------------------

def feature_mean(matrix):
    ret = np.mean(matrix, axis=0).flatten()
    names = ['mean_' + str(i) for i in range(matrix.shape[1])]
    return ret, names

def feature_mean_d(h1, h2):
    ret = (feature_mean(h2)[0] - feature_mean(h1)[0]).flatten()
    names = ['mean_d_h2h1_' + str(i) for i in range(h1.shape[1])]
    return ret, names

def feature_mean_q(q1, q2, q3, q4):
    v1, v2, v3, v4 = [feature_mean(q)[0] for q in [q1, q2, q3, q4]]
    ret = np.hstack([v1, v2, v3, v4, v1 - v2, v1 - v3, v1 - v4, v2 - v3, v2 - v4, v3 - v4]).flatten()
    names = []
    for i in range(4): names.extend(['mean_q' + str(i+1) + "_" + str(j) for j in range(len(v1))])
    for i in range(3):
        for j in range(i+1, 4):
            names.extend(['mean_d_q' + str(i+1) + 'q' + str(j+1) + "_" + str(k) for k in range(len(v1))])
    return ret, names

def feature_stddev(matrix):
    # FIX: Add check for sufficient samples for stddev
    if matrix.shape[0] < 2: 
        # print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: WARNING: Not enough samples ({matrix.shape[0]}) for stddev. Returning NaNs.")
        num_cols = matrix.shape[1] if matrix.ndim > 1 else 0
        return np.full(num_cols, np.nan), ['std_' + str(i) for i in range(num_cols)]
    ret = np.std(matrix, axis=0, ddof=1).flatten()
    names = ['std_' + str(i) for i in range(matrix.shape[1])]
    return ret, names

def feature_stddev_d(h1, h2):
    ret = (feature_stddev(h2)[0] - feature_stddev(h1)[0]).flatten()
    names = ['std_d_h2h1_' + str(i) for i in range(h1.shape[1])]
    return ret, names

def feature_moments(matrix):
    # FIX: Add check for sufficient samples for moments
    if matrix.shape[0] < 4: # Skewness needs at least 2, Kurtosis needs at least 4. Min is 4.
        # print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: WARNING: Not enough samples ({matrix.shape[0]}) for moments. Returning NaNs.")
        num_cols = matrix.shape[1] if matrix.ndim > 1 else 0
        skw = np.full(num_cols, np.nan)
        krt = np.full(num_cols, np.nan)
        ret = np.append(skw, krt)
        names = ['skew_' + str(i) for i in range(num_cols)] + ['kurt_' + str(i) for i in range(num_cols)]
        return ret, names
    skw = scipy.stats.skew(matrix, axis=0, bias=False)
    krt = scipy.stats.kurtosis(matrix, axis=0, bias=False)
    ret = np.append(skw, krt)
    names = ['skew_' + str(i) for i in range(matrix.shape[1])] + ['kurt_' + str(i) for i in range(matrix.shape[1])]
    return ret, names

def feature_max(matrix):
    ret = np.max(matrix, axis=0).flatten()
    names = ['max_' + str(i) for i in range(matrix.shape[1])]
    return ret, names

def feature_max_d(h1, h2):
    ret = (feature_max(h2)[0] - feature_max(h1)[0]).flatten()
    names = ['max_d_h2h1_' + str(i) for i in range(h1.shape[1])]
    return ret, names

def feature_max_q(q1, q2, q3, q4):
    v1, v2, v3, v4 = [feature_max(q)[0] for q in [q1, q2, q3, q4]]
    ret = np.hstack([v1, v2, v3, v4, v1 - v2, v1 - v3, v1 - v4, v2 - v3, v2 - v4, v3 - v4]).flatten()
    names = []
    for i in range(4): names.extend(['max_q' + str(i+1) + "_" + str(j) for j in range(len(v1))])
    for i in range(3):
        for j in range(i+1, 4):
            names.extend(['max_d_q' + str(i+1) + 'q' + str(j+1) + "_" + str(k) for k in range(len(v1))])
    return ret, names

def feature_min(matrix):
    ret = np.min(matrix, axis=0).flatten()
    names = ['min_' + str(i) for i in range(matrix.shape[1])]
    return ret, names

def feature_min_d(h1, h2):
    ret = (feature_min(h2)[0] - feature_min(h1)[0]).flatten()
    names = ['min_d_h2h1_' + str(i) for i in range(h1.shape[1])]
    return ret, names

def feature_min_q(q1, q2, q3, q4):
    v1, v2, v3, v4 = [feature_min(q)[0] for q in [q1, q2, q3, q4]]
    ret = np.hstack([v1, v2, v3, v4, v1 - v2, v1 - v3, v1 - v4, v2 - v3, v2 - v4, v3 - v4]).flatten()
    names = []
    for i in range(4): names.extend(['min_q' + str(i+1) + "_" + str(j) for j in range(len(v1))])
    for i in range(3):
        for j in range(i+1, 4):
            names.extend(['min_d_q' + str(i+1) + 'q' + str(j+1) + "_" + str(k) for k in range(len(v1))])
    return ret, names

def feature_covariance_matrix(matrix):
    # FIX: Add check for sufficient samples for covariance matrix
    if matrix.shape[0] < 2: 
        # print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: WARNING: Not enough samples ({matrix.shape[0]}) for covariance matrix. Returning NaNs.")
        num_cols = matrix.shape[1] if matrix.ndim > 1 else 0
        return np.full(int(num_cols * (num_cols + 1) / 2), np.nan), [], np.full((num_cols, num_cols), np.nan)
    covM = np.cov(matrix.T)
    indx = np.triu_indices(covM.shape[0])
    ret = covM[indx]
    names = ['covM_' + str(i) + '_' + str(j) for i, j in zip(*indx)]
    return ret, names, covM

def feature_eigenvalues(covM):
    # FIX: Add check for valid covariance matrix and error handling for eigvals
    if covM.ndim < 2 or covM.shape[0] != covM.shape[1] or covM.shape[0] == 0 or np.isnan(covM).any(): 
        # print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: WARNING: Invalid/empty covariance matrix for eigenvalues: {covM.shape}. Returning NaNs.")
        num_cols = covM.shape[0] if covM.ndim > 1 else 0
        return np.full(num_cols, np.nan), []
    
    try:
        ret = np.linalg.eigvals(covM).flatten()
    except np.linalg.LinAlgError as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: WARNING: LinAlgError in eigvals: {e}. Covariance matrix might be singular. Returning NaNs.")
        num_cols = covM.shape[0]
        return np.full(num_cols, np.nan), []
    
    names = ['eigenval_' + str(i) for i in range(covM.shape[0])]
    return ret, names

def feature_logcov(covM):
    # FIX: Add check for valid covariance matrix and error handling for logm
    if covM.ndim < 2 or covM.shape[0] != covM.shape[1] or covM.shape[0] == 0 or np.isnan(covM).any():
        # print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: WARNING: Invalid/empty covariance matrix for logcov: {covM.shape}. Returning NaNs.")
        num_cols = covM.shape[0] if covM.ndim > 1 else 0
        indx = np.triu_indices(num_cols)
        names = ['logcovM_' + str(i) + '_' + str(j) for i, j in zip(*indx)]
        return np.full(len(names), np.nan), names, np.full((num_cols, num_cols), np.nan) 

    try:
        log_cov = scipy.linalg.logm(covM)
        indx = np.triu_indices(log_cov.shape[0])
        ret = np.abs(log_cov[indx])
        
        names = []
        for i in np.arange(0, log_cov.shape[1]):
            for j in np.arange(i, log_cov.shape[1]):
                names.extend(['logcovM_' + str(i) + '_' + str(j)])
        
        return ret, names, log_cov
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: ERROR during logm calculation: {e}. Returning NaNs for logcov.")
        num_cols = covM.shape[0] if covM.ndim > 1 else 0
        indx = np.triu_indices(num_cols)
        names = ['logcovM_' + str(i) + '_' + str(j) for i, j in zip(*indx)]
        return np.full(len(names), np.nan), names, np.full((num_cols, num_cols), np.nan)


def feature_fft(matrix, period=1., mains_f=50., filter_mains=True, filter_DC=True, normalise_signals=True, ntop=10, get_power_spectrum=True):
    N = matrix.shape[0]
    if N == 0:
        # print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: WARNING: Empty matrix for FFT. Returning empty.")
        return np.array([]), []
    
    if matrix.ndim == 1: # If only one channel, make it 2D
        matrix = matrix.reshape(-1, 1)

    T = period / N
    if normalise_signals:
        max_val = np.max(matrix, axis=0)
        min_val = np.min(matrix, axis=0)
        range_val = max_val - min_val
        # Avoid division by zero: if range is 0, set normalized value to 0
        normalized_matrix = np.where(range_val == 0, 0.0, -1 + 2 * (matrix - min_val) / range_val)
        matrix = normalized_matrix
            
    try:
        fft_values = np.abs(scipy.fft.fft(matrix, axis=0))[0:N//2] * 2 / N
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: ERROR during FFT calculation: {e}. Returning empty.")
        return np.array([]), []

    freqs = np.linspace(0.0, 1.0 / (2.0 * T), N//2)
    
    if freqs.size == 0 or fft_values.size == 0:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: WARNING: Empty freqs or fft_values after initial FFT. Returning empty.")
        return np.array([]), []

    if filter_DC:
        if fft_values.shape[0] > 1: # Ensure slicing is valid (more than 1 element)
            fft_values, freqs = fft_values[1:], freqs[1:]
        else:
            # print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: WARNING: Too few samples for DC filter. Skipping filter.")
            pass
    
    if filter_mains and freqs.size > 0:
        indx = np.where(np.abs(freqs - mains_f) <= 1)
        if indx[0].size > 0: # Check if any mains frequencies were found
            fft_values, freqs = np.delete(fft_values, indx, axis=0), np.delete(freqs, indx)
        else:
            # print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: No mains frequencies to filter.")
            pass
    
    if freqs.size == 0 or fft_values.size == 0:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: WARNING: Empty freqs or fft_values after filtering. Returning empty.")
        return np.array([]), []

    ntop_actual = min(ntop, freqs.shape[0])
    if ntop_actual == 0:
        # print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: WARNING: No frequencies left for topFreq. Returning empty.")
        if get_power_spectrum:
            return fft_values.flatten(order='F'), ['fft' + "{:03d}".format(int(j)) + "_" + str(i) for i in range(fft_values.shape[1]) for j in 10 * np.round(freqs, 1)] if freqs.size > 0 else []
        return np.array([]), []
        
    indx = np.argsort(fft_values, axis=0)[::-1][:ntop_actual]
    ret = freqs[indx].flatten(order='F')
    names = ['topFreq_' + str(j) + "_" + str(i) for i in range(fft_values.shape[1]) for j in range(1, ntop_actual + 1)]

    if get_power_spectrum:
        ret = np.hstack([ret, fft_values.flatten(order='F')])
        if freqs.size > 0:
            names += ['fft' + "{:03d}".format(int(j)) + "_" + str(i) for i in range(fft_values.shape[1]) for j in 10 * np.round(freqs, 1)]
        else:
            names += ['fft_empty_channel_' + str(i) for i in range(fft_values.shape[1])]

    return ret, names

# --------------------- High-Level Feature Vector Extraction ---------------------

def calc_feature_vector(matrix, state):
    # Ensure matrix has enough samples for splitting
    if matrix.shape[0] < 4: # Need at least 4 samples for q1,q2,q3,q4 and h1/h2
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: WARNING: Not enough samples ({matrix.shape[0]}) for splitting in calc_feature_vector. Returning None, None.")
        return None, None
    
    # Ensure matrix has enough columns for channels
    if matrix.ndim < 2 or matrix.shape[1] < 1:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: WARNING: Invalid matrix shape for calc_feature_vector: {matrix.shape}. Expected 2D array with channels. Returning None, None.")
        return None, None

    h1, h2 = np.split(matrix, [matrix.shape[0] // 2])
    q1, q2, q3, q4 = np.split(matrix, [int(0.25 * matrix.shape[0]), int(0.5 * matrix.shape[0]), int(0.75 * matrix.shape[0])])
    
    # Check if any split results in empty arrays
    if any(q.shape[0] == 0 for q in [h1, h2, q1, q2, q3, q4]):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: WARNING: Empty splits in calc_feature_vector. Shapes: h1={h1.shape}, h2={h2.shape}, q1={q1.shape}, q2={q2.shape}, q3={q3.shape}, q4={q4.shape}. Returning None, None.")
        return None, None

    var_names = []
    var_values = np.array([])

    feature_functions_with_args = [
        (feature_mean, (matrix,)),
        (feature_mean_d, (h1, h2)),
        (feature_mean_q, (q1, q2, q3, q4)),
        (feature_stddev, (matrix,)), # FIX: Corrected argument passing
        (feature_stddev_d, (h1, h2)),
        (feature_moments, (matrix,)), # FIX: Corrected argument passing
        (feature_max, (matrix,)),
        (feature_max_d, (h1, h2)),
        (feature_max_q, (q1, q2, q3, q4)),
        (feature_min, (matrix,)),
        (feature_min_d, (h1, h2)),
        (feature_min_q, (q1, q2, q3, q4))
    ]

    for func, args in feature_functions_with_args:
        try:
            x, v = func(*args)
            if x.size == 0:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: WARNING: Feature function {func.__name__} returned empty array. Skipping.")
                continue
            var_names.extend(v)
            var_values = np.hstack([var_values, x])
        except Exception as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: ERROR calculating feature {func.__name__}: {e}. Skipping this feature.")
            pass

    # Handle covariance matrix and related features
    covM_ret, covM_names, covM = feature_covariance_matrix(matrix)
    if covM_ret.size > 0:
        var_names.extend(covM_names)
        var_values = np.hstack([var_values, covM_ret])

        eig_ret, eig_names = feature_eigenvalues(covM)
        if eig_ret.size > 0:
            var_names.extend(eig_names)
            var_values = np.hstack([var_values, eig_ret])

        logcov_ret, logcov_names, _ = feature_logcov(covM)
        if logcov_ret.size > 0:
            var_names.extend(logcov_names)
            var_values = np.hstack([var_values, logcov_ret])
    else:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: WARNING: Covariance matrix could not be calculated. Skipping related features.")

    fft_ret, fft_names = feature_fft(matrix, period=1.)
    if fft_ret.size > 0:
        var_names.extend(fft_names)
        var_values = np.hstack([var_values, fft_ret])
    else:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: WARNING: FFT features could not be calculated. Skipping.")

    if state is not None:
        var_values = np.hstack([var_values, np.array([state])])
        var_names.append('Label')

    if var_values.size == 0 or len(var_names) == 0:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: WARNING: No features or names generated by calc_feature_vector. Returning None, None.")
        return None, None

    return var_values, var_names

# --------------------- Original Full Feature Vector Generator (ADAPTED FOR REAL-TIME SINGLE CHUNK) ---------------------
# This function is now designed to take a single file (1-second data) and produce 486 features
# by simulating the 'lagged' component.
# This will be the generate_feature_vectors_from_samples directly imported by Streamlit.
def generate_feature_vectors_from_samples(file_path, nsamples, period, state=None, remove_redundant=True, cols_to_ignore=None):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: (Real-Time Gen) Entering generate_feature_vectors_from_samples for file: {file_path}")
    matrix = matrix_from_csv_file(file_path)
    if matrix.size == 0:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: (Real-Time Gen) matrix_from_csv_file returned empty matrix. Cannot proceed.")
        return None, None

    r = None
    headers = None 
    
    try:
        # Step 1: Extract EEG data columns (skipping timestamp)
        if cols_to_ignore is not None and len(cols_to_ignore) > 0:
            all_cols = np.arange(matrix.shape[1])
            cols_to_keep = np.setdiff1d(all_cols, cols_to_ignore)
            s = matrix[:, cols_to_keep].astype(float) # Ensure EEG data is float
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: (Real-Time Gen) Columns ignored. 's' (EEG data) shape: {s.shape}")
        else:
            s = matrix[:, 1:].astype(float) # Default to skipping column 0 (timestamp), ensure float
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: (Real-Time Gen) Using columns from index 1. 's' (EEG data) shape: {s.shape}")

        if s.shape[0] == 0 or s.shape[1] < 2: # Ensure we have data rows and at least 2 channels (AF7, TP9)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: (Real-Time Gen) Insufficient data columns/rows after selecting EEG channels for resampling: {s.shape}. Returning None, None.")
            return None, None

        # Step 2: Resample the EEG data (s)
        time_base = matrix[:, 0].flatten().astype(float) # Ensure time_base is float and 1D
        if time_base.size != matrix.shape[0]:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: (Real-Time Gen) ERROR: Time base size mismatch for resampling. Cannot proceed.")
            return None, None

        ry, _ = scipy.signal.resample(s, num=nsamples, t=time_base, axis=0)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: (Real-Time Gen) Data resampled to {nsamples} samples. 'ry' shape: {ry.shape}")

        # Step 3: Calculate features for this resampled data (r will be ~252 features)
        r, headers = calc_feature_vector(ry, state)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: (Real-Time Gen) calc_feature_vector returned features (size: {r.size if r is not None else 'None'}) and headers (count: {len(headers) if headers is not None else 'None'}).")

        if r is None or headers is None or r.size == 0:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: (Real-Time Gen) calc_feature_vector returned no valid data/headers. Returning None, None.")
            return None, None
        
        # --- CRUCIAL ADAPTATION FOR REAL-TIME TO GET 486 FEATURES ---
        # Your model expects 486 features, but calc_feature_vector on a single window (150 samples)
        # produces ~252 features. The original training logic stacked previous_vector (234 features) + current_r (252 features).
        # We need to simulate this. The simplest hack is to take the current 'r' (252 features)
        # and create a "dummy" previous_vector from it or pad it.
        # A common (though not perfect) way to handle this when a true 'lag' is unavailable
        # for a single window is to duplicate features or create a zero/mean padded previous vector.
        # Given 486 = 252 (current) + 234 (lagged), we need 234 features for the lagged part.
        # We'll take the first 234 features of 'r' (if 'r' is long enough) as the 'lagged' part.
        
        expected_lagged_features_count = 486 - r.size # Should be 486 - 252 = 234

        if r.size >= expected_lagged_features_count:
            # Take the first 'expected_lagged_features_count' features from current 'r' as the lagged part
            previous_vector_simulated = r[:expected_lagged_features_count]
            # Construct the full feature vector as [simulated_previous_vector, current_r]
            ret = np.hstack([previous_vector_simulated, r])
            
            # Adjust headers for this simulated lagged feature vector
            simulated_lagged_headers = ["lag1_" + s for s in headers[:expected_lagged_features_count]]
            headers = simulated_lagged_headers + list(headers) # Combine lagged headers with current headers
            
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: (Real-Time Gen) Successfully simulated lagged features. Final raw feature count: {ret.size}.")

        else:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: (Real-Time Gen) WARNING: Current 'r' size ({r.size}) is too small to simulate {expected_lagged_features_count} lagged features. Model input size will be incorrect. Attempting to use 'r' directly, which will lead to shape mismatch.")
            ret = r # Fallback: use r directly, will cause model mismatch
            headers = list(headers) # Ensure headers is a list

    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: (Real-Time Gen) ERROR during single-slice feature generation: {e}. Returning None, None.")
        return None, None 

    # Step 4: Handle redundant features (if requested and features exist)
    if ret is not None and headers is not None and remove_redundant:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: (Real-Time Gen) Removing redundant features...")
        feat_names = list(headers) # Create a mutable copy of the headers
        
        # Determine number of channels (assumed 2: AF7, TP9)
        # This part depends on the structure of feature_fft and other functions' names
        num_channels = 2 
        
        indices_to_delete = []
        for to_rm_prefix in ["lag1_mean_q3_", "lag1_mean_q4_", "lag1_mean_d_q3q4_",
                             "lag1_max_q3_", "lag1_max_q4_", "lag1_max_d_q3q4_",
                             "lag1_min_q3_", "lag1_min_q4_", "lag1_min_d_q3q4_"]:
            for j in range(num_channels):
                rm_str = to_rm_prefix + str(j)
                if rm_str in feat_names:
                    try:
                        idx = feat_names.index(rm_str)
                        indices_to_delete.append(idx)
                    except ValueError:
                        pass # Should not happen if 'if rm_str in feat_names' is true
        
        # Sort indices in descending order to avoid issues when popping/deleting
        indices_to_delete.sort(reverse=True)
        
        for idx in indices_to_delete:
            if feat_names: # Check if list is not empty before popping
                feat_names.pop(idx)
            
            if ret.ndim == 1: ret = ret.reshape(1, -1) # Make it (1, num_features)
            if ret.shape[1] > idx: # Ensure index is valid for deletion
                ret = np.delete(ret, idx, axis=1)
            else:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: (Real-Time Gen) WARNING: Index {idx} out of bounds for ret.shape[1]={ret.shape[1]} during removal of a redundant feature. Skipping delete for this feature.")
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: (Real-Time Gen) Redundant feature removal complete. Final features shape: {ret.shape if ret is not None else 'None'}.")
        return ret, feat_names
    
    elif ret is not None and headers is not None:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: (Real-Time Gen) Returning features without redundant removal (ret shape: {ret.shape}).")
        return ret, list(headers) # Ensure headers is a list
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: (Real-Time Gen) No features or headers to return after processing. Returning None, None.")
    return None, None

# --------------------- Final Matrix Assembly (Standard Training Data Generation) ---------------------
# This function is used for generating the training dataset, not for real-time inference.
# It uses the original generate_feature_vectors_from_samples logic (which has the while loop and previous_vector)
# and maintains nsamples=150 as per your model's training.
def gen_training_matrix(directory_path, output_file, cols_to_ignore):
    # print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: (Training Mode) Entering gen_training_matrix for directory: {directory_path}")
    FINAL_MATRIX = None
    header = None # Initialize header for safety

    for x in os.listdir(directory_path):
        if not x.lower().endswith('.csv'):
            continue
        if 'test' in x.lower(): # Skip test files if found in dataset directory
            continue
        try:
            name, state_str, _ = x[:-4].split('-') # name, state_string
        except ValueError:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: (Training Mode) Wrong file name format: {x}. Skipping.")
            continue

        state = {'positive': 2., 'neutral': 1., 'negative': 0.}.get(state_str.lower(), None)
        if state is None:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: (Training Mode) Unknown state in file name: {x}. Skipping.")
            continue
            
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: (Training Mode) Using file {x}")
        full_file_path = os.path.join(directory_path, x)
        
        # This calls the original generate_feature_vectors_from_samples (the multi-slice version)
        # for training data generation with nsamples=150.
        # NOTE: This is your original generate_feature_vectors_from_samples function from the top of this file.
        vectors, current_header = _original_generate_feature_vectors_multi_slice( # Renaming to prevent conflict
            file_path=full_file_path, 
            nsamples=150, # IMPORTANT: Keep this at 150 for training data consistency
            period=1.,
            state=state,
            remove_redundant=True,
            cols_to_ignore=cols_to_ignore
        )
        
        if vectors is not None and current_header is not None:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: (Training Mode) Resulting vector shape for file {x}: {vectors.shape}")
            if FINAL_MATRIX is None:
                FINAL_MATRIX = vectors
                header = current_header # Capture header from the first successful file
            else:
                FINAL_MATRIX = np.vstack([FINAL_MATRIX, vectors])
        else:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: (Training Mode) Skipping file {x} due to no valid feature vectors generated.")

    if FINAL_MATRIX is not None:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: (Training Mode) FINAL_MATRIX shape: {FINAL_MATRIX.shape}")
        np.random.shuffle(FINAL_MATRIX)
        
        if header is not None:
            np.savetxt(output_file, FINAL_MATRIX, delimiter=',', header=','.join(header), comments='')
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: (Training Mode) Final training matrix saved to {output_file} with header.")
        else:
            np.savetxt(output_file, FINAL_MATRIX, delimiter=',', comments='')
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: (Training Mode) Final training matrix saved to {output_file} WITHOUT header (no header found).")
    else:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: (Training Mode) No final matrix generated. All files failed or directory was empty.")

    return None

# --------------------- Original Full Feature Vector Generator (ROBUSTIFIED for multi-slice processing) ---------------------
# This is a robustified version of your original generate_feature_vectors_from_samples
# It will be called specifically by gen_training_matrix.
def _original_generate_feature_vectors_multi_slice(file_path, nsamples, period, 
                                           state = None, 
                                           remove_redundant = True,
                                           cols_to_ignore = None):

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: (Multi-Slice Gen) Entering _original_generate_feature_vectors_multi_slice for file: {file_path}")
    matrix = matrix_from_csv_file(file_path)
    if matrix.size == 0:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: (Multi-Slice Gen) Empty matrix from CSV. Returning None, None.")
        return None, None
    
    t = 0.
    previous_vector = None
    ret = None
    feat_names = [] # Initialize feat_names here

    while True:
        try:
            s, dur = get_time_slice(matrix, start = t, period = period)
            if cols_to_ignore is not None:
                if s.size == 0:
                    break # No data left after slicing
                s = np.delete(s, cols_to_ignore, axis = 1).astype(float) # Ensure EEG data is float
        except IndexError:
            # print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: (Multi-Slice Gen) IndexError during get_time_slice. Breaking loop.")
            break
        except Exception as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: (Multi-Slice Gen) Error during get_time_slice or column deletion: {e}. Breaking loop.")
            break

        if s.shape[0] == 0 or dur < 0.9 * period:
            # print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: (Multi-Slice Gen) Slice too short or empty. Breaking loop. Shape={s.shape}, Duration={dur}")
            break
        
        if s.shape[1] < 2: # Ensure 's' has at least 2 channels for resampling (AF7, TP9)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: (Multi-Slice Gen) Insufficient columns in slice 's' ({s.shape[1]}) for resampling. Skipping slice.")
            t += 0.5 * period # Advance to next slice
            continue # Skip to next iteration

        try:
            ry, rx = scipy.signal.resample(s, num = nsamples, t = np.linspace(0, period, s.shape[0]), axis = 0) # Use simple linspace for time_base
            # print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: (Multi-Slice Gen) Slice resampled. ry shape: {ry.shape}")
        except Exception as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: (Multi-Slice Gen) Error during resampling: {e}. Skipping this slice.")
            t += 0.5 * period # Advance to next slice
            continue

        r, current_headers = calc_feature_vector(ry, state)
        
        if r is None or current_headers is None:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: (Multi-Slice Gen) calc_feature_vector returned None. Skipping this slice.")
            t += 0.5 * period # Advance to next slice
            continue

        if not feat_names: # Only set headers once from the first successful feature vector
            feat_names = current_headers
            # print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: (Multi-Slice Gen) Headers initialized from first vector. Count: {len(feat_names)}")
        
        if previous_vector is not None:
            try:
                feature_vector = np.hstack([previous_vector, r])
                if ret is None:
                    ret = feature_vector
                else:
                    ret = np.vstack([ret, feature_vector])
                # print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: (Multi-Slice Gen) Features concatenated. Ret shape: {ret.shape}")
            except ValueError as ve:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: (Multi-Slice Gen) Shape mismatch during hstack: {ve}. Skipping this feature vector.")
            except Exception as e:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: (Multi-Slice Gen) Error stacking features: {e}. Skipping this feature vector.")
        
        # Store the vector of the previous window (remove label if state is present)
        previous_vector_for_next_iter = r
        if state is not None:
            previous_vector_for_next_iter = previous_vector_for_next_iter[:-1]
        previous_vector = previous_vector_for_next_iter
        
        t += 0.5 * period # Move to the next slice (overlapping by 0.5 period)


    if ret is None:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: (Multi-Slice Gen) No valid feature vectors generated. Returning None, None.")
        return None, None
    
    # Ensure feat_names is properly set if ret is not None (fallback)
    if not feat_names and ret is not None:
        if r is not None and current_headers is not None:
            feat_names = current_headers
            if state is not None:
                feat_names = feat_names[:-1]

        if not feat_names and ret is not None:
             print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: (Multi-Slice Gen) Still no headers. Creating generic headers for {ret.shape[1]} features. This may cause issues.")
             feat_names = [f"feature_{i}" for i in range(ret.shape[1])]


    if remove_redundant and feat_names and ret is not None:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: (Multi-Slice Gen) Removing redundant features...")
        
        # Assuming ry.shape[1] is 2 (AF7, TP9) based on previous logs from your original code
        num_channels = 2 
        
        indices_to_delete = []
        for to_rm_prefix in ["lag1_mean_q3_", "lag1_mean_q4_", "lag1_mean_d_q3q4_",
                             "lag1_max_q3_", "lag1_max_q4_", "lag1_max_d_q3q4_",
                             "lag1_min_q3_", "lag1_min_q4_", "lag1_min_d_q3q4_"]:
            for j in range(num_channels):
                rm_str = to_rm_prefix + str(j)
                if rm_str in feat_names:
                    try:
                        idx = feat_names.index(rm_str)
                        indices_to_delete.append(idx)
                    except ValueError: # Should not happen if 'if rm_str in feat_names' is true
                        pass 

        indices_to_delete.sort(reverse=True) # Sort descending for safe popping/deletion
        
        for idx in indices_to_delete:
            if feat_names: # Check if list is not empty before popping
                feat_names.pop(idx)
            
            if ret.ndim == 1: ret = ret.reshape(1, -1) # Make it 2D if only one row
            if ret.shape[1] > idx: # Ensure index is valid for deletion
                ret = np.delete(ret, idx, axis=1)
            else:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: (Multi-Slice Gen) WARNING: Index {idx} out of bounds for ret.shape[1]={ret.shape[1]} during removal of a redundant feature. Skipping delete for this feature.")
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FE: (Multi-Slice Gen) Redundant feature removal complete. Final features shape: {ret.shape}.")

    return ret, feat_names

# --------------------- Main Entry Point ---------------------
if __name__ == '__main__':
    # This block is only executed when eeg_feature_extractor.py is run directly.
    # It uses the _original_generate_feature_vectors_multi_slice function for training data generation.

    # Hardcoded parameters (from your original __main__ block)
    directory_path = r"C:\MY Laptop\Emotion Detection\arduino" 
    output_file = r"C:\MY Laptop\Emotion Detection\data.csv" 
    cols_to_ignore = None 

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running eeg_feature_extractor.py as main script (for training data generation).")
    gen_training_matrix(directory_path, output_file, cols_to_ignore)