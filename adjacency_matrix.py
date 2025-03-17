import numpy as np

def create_trialwise_adjacency_matrices(subj1_data, subj2_data, percentile=55):
    """
    Create adjacency matrices for each trial using Pearson correlation.
    """
    num_trials, num_channels = subj1_data.shape[0], subj1_data.shape[1]
    adjacency_matrices = []

    for trial in range(num_trials):
        subj1_trial, subj2_trial = subj1_data[trial], subj2_data[trial]
        correlation_matrix = np.array([[np.corrcoef(subj1_trial[i], subj2_trial[j])[0, 1] 
                                        for j in range(num_channels)] 
                                        for i in range(num_channels)])
        threshold = np.percentile(abs(correlation_matrix), percentile)
        adjacency_matrix = np.where(abs(correlation_matrix) >= threshold, 1, 0)
        adjacency_matrices.append(adjacency_matrix)

    return adjacency_matrices
