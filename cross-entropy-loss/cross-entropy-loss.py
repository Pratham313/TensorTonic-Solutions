import numpy as np

def cross_entropy_loss(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Get predicted probability for the correct class for each sample
    correct_probs = y_pred[np.arange(len(y_true)), y_true]
    
    # Compute average cross-entropy loss
    return -np.mean(np.log(correct_probs))