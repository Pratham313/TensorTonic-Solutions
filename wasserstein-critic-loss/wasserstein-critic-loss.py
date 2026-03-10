import numpy as np

def wasserstein_critic_loss(real_scores, fake_scores):
    return np.mean(fake_scores) - np.mean(real_scores)