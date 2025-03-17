from scipy.io import loadmat
import numpy as np

def load_eeg_data(file_paths):
    """
    Load EEG data from multiple .mat files and concatenate them.
    """
    data_list = []
    for path in file_paths:
        data = loadmat(path)['data']
        data_list.append(data[:30, :, :])  # First 30 channels
        data_list.append(data[30:, :, :])  # Remaining channels

    combined_data = np.concatenate(data_list, axis=2)
    return combined_data.reshape(combined_data.shape[2], combined_data.shape[0], combined_data.shape[1])

def load_labels(label_paths):
    """
    Load EEG labels from multiple .mat files and concatenate them.
    """
    labels = [loadmat(path)['Labels'].T for path in label_paths]
    return np.concatenate(labels, axis=0)
