from scipy.stats import norm as dist_model
from numpy import round, max
import numpy as np

   
def fit_distribution_single_class(y_proba):
    """Fits a list of numbers under a Gaussian distribution
    using both existing samples and mirrored versions of the
    existing numbers with respect to 1.
    
    Parameters
    ----------
    y_proba : np.ndarray
        array of numbers to fit under Gaussian distribution
    
    Returns
    -------
    pos_mu : float
        mean of the distribution, should always be 1
    pos_std : float
        standard deviation of the probability distribution
    """
    prob_pos = [p for p in y_proba] + [2 - p for p in y_proba]
    pos_mu, pos_std = dist_model.fit(prob_pos)
    return pos_mu, pos_std, prob_pos


def compute_threshold(mu_stds, alpha=3):
    """Compute the thresholds for each class in order to perform
    outlier detection for entry rejection
    
    Parameters
    ----------
    mu_stds : list
        list of standard deviations
    alpha : int, optional
        number of standard deviations to consider a data point
        as an outlier, by default 3
    
    Returns
    -------
    class_thresholds : list
        list of every threshold for every class
    """
    class_thresholds = [max([0.5, 1 - alpha * std[1]]) for std in mu_stds]
    return round(class_thresholds, 4)
