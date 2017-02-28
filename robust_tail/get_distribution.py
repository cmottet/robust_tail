import numpy as np
import pandas as pd

def get_distribution(mu, sigma, x1=None):
    """
    Returns a 2-point support feasible distribution function for program (5)

    Returns a 2-point support feasible distribution function for program (5), i.e. a distribution function with
    two point masses (p1,p2) and supports (x1,x2) satisfying the three equations p1 + p2 = 1,  p1x1 + p2x2 = mu,
    and m1x1^2 + p2x2^2 = sigma.

    :param mu: scalar
        First moment of the distribution function
    :param sigma: number
        Second moment of the distribution function
    :param x1: scalar
        First point support of the distribution. Comprised between 0 and mu. Default value is None, in which case,
        x1 is drawn from a uniform distribution over the interval [0,mu].

    :return: a data frame
        Data frame containing two columns :
        * x : the point supports
        * p : the probability masses
    """

    if mu**2 > sigma:
        output = None

    if mu**2 == sigma:
        output = np.dataframe({'x': [mu], 'p': [1]})

    if mu**2 < sigma:
        if x1 is None:
            x1 = runif(1,0,mu)

        x2 = mu + (sigma - mu**2)/(mu-x1)
        p1 = (sigma - mu**2)/((sigma - mu**2) + (x1-mu)**2)
        p2 = 1 - p1

        output = pd.DataFrame({'x': [x1, x2], 'p': [p1, p2]})

    return output

