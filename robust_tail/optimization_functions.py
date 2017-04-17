from math import sqrt
import numpy as np
from .W import W
from .get_distribution import get_distribution
from .gensa_modified import _gensa_modified

# ' git clone https://github.com/sgubianpm/pygensa.git
# ' import pip
# ' pip.main(["install","--user", "--upgrade","/Users/cmottet/Packages/python/pygensa"])
# ' _all__ = ['compute_bound']


direction_type = ['max', 'min']


def _compute_bound_val(H,  mu, sigma, limsup, direction='max'):
    """
    Solves programs (5) over 2-point masses distribution functions, when nu = 1.
    """
    if direction not in direction_type:
        raise ValueError("Invalid direction. Expected one of: %s" % direction_type)

    if isinstance(mu, np.ndarray):
        mu = np.assacalar(mu)

    if isinstance(sigma, np.ndarray):
        sigma = np.assacalar(sigma)

    scale = (1 if direction == 'min' else -1)

    # Exclusion of trivial scenarios
    if mu**2 > sigma:
        output = {'bound': scale*float('inf'), 'P': None}

    if mu**2 == sigma:
        output = {'bound': H(mu), 'P': get_distribution(mu, sigma)}

    # Treatment of non-trivial scenarios
    if mu**2 < sigma:
        z = _gensa_modified(func=lambda x1: scale*W(x1, mu, sigma, limsup, H), x0=None, bounds=[[0, mu]])
        output = {'bound': scale*z.fun, 'P': get_distribution(mu, sigma, x1=z.x[0])}

    return output


#
# Solves programs (EC.19) over 2-point masses distribution functions, when nu = 1.
#
def _compute_bound_int(H, mu, sigma, limsup, direction="max"):

    if direction not in direction_type:
        raise ValueError("Invalid direction. Expected one of: %s" % direction_type)

    scale = (1 if direction == 'min' else -1)

    # Exclude trivial scenarios
    if mu[0] > mu[1] or sigma[0] > sigma[1]:
        output = {'bound': scale * float('inf'), 'P': None}

    if mu[0]**2 > sigma[1]:
        output = {'bound': scale * float('inf'), 'P': None}

    if mu[0]**2 == sigma[1]:
        output = {'bound': H(mu[0]), 'P': get_distribution(mu[0], sigma[1])}

    # Non-trivial scenario
    if mu[0]**2 < sigma[1]:
        # First Subprogram
        lower = [0, max(sigma[0], mu[1]**2)]
        upper = [mu[1], sigma[1]]

        if np.greater(lower, upper).any():
            output = {'bound': scale * float('inf'), 'P1': None}
        else:
            z1 = _gensa_modified(func=lambda args: scale*W(x=args[0], w=mu[1], rho=args[1], limsup=limsup, H=H),
                                 x0=lower,
                                 bounds=[[lower[i], upper[i]] for i in [0, 1]])

            P1 = get_distribution(mu=mu[1], sigma=z1.x[1], x1=z1.x[0])

        # Second Subprogram
        lower = [0, mu[0]]
        upper = np.repeat(min(mu[1], sqrt(sigma[1])), 2)

        if np.greater(lower, upper).any():
            output = {'bound': scale * float('inf'), 'P2': None}
        else:
            f2 = lambda args: np.where(args[0] > args[1], float('inf'), scale*W(x=args[0], w=args[1], rho=sigma[1], limsup=limsup, H=H))
            z2 = _gensa_modified(func=f2,
                                 x0=lower,
                                 bounds=[[lower[i], upper[i]] for i in range(2)])

            P2 = get_distribution(mu=z2.x[1], sigma=sigma[1], x1=z2.x[0])

        bound = max(scale*z1.fun, scale*z2.fun)
        P = (P1 if bound == scale*z1.fun else P2)

        output = {'bound': bound, 'P': P}

    return output


def compute_bound(H, mu, sigma, limsup, nu=1, direction="max"):
    """
    Solves programs (5) and (EC.19) over 2-point masses distribution functions

    This function solves programs (5) and (EC.19), in the case
    when their respective feasible region is restricted to distribution functions with at most two point supports.

    :param H: The function defined in the objective value of program  (5) and (EC.19)
    :param mu: Either an ordered numpy.ndarray containing bounds of the first moment in program (EC.19), or a scalar  as in program
    (5)
    :param sigma: Either an ordered numpy.ndarray containing bounds of the second moment in program (EC.19) or a scalar as in
    program (5)
    :param limsup: A real number  giving the limit superior value of the ratio H(x)/x^2 when x goes to infinity
    :param nu: A real number as defined in program (5) and  (EC.19) (actually denoted nubar in the latter case)
    :param direction:  A string either "min" or "max" identifying the type of whether program (5) should be a min or a
     max program. Default is "max".
    :return: A dictionnary with the worst-case bound in 'bound' and the optimal distribution function in P

    :Example:
    ##############################################################################
    #### Solving a program alike  to (5)
    ##############################################################################
    ####
    #### max P(X > c)
    #### s.t. sum(px) = 1
    ####      sum(px**2) = 2
    ####      (p,x) is a two point mass distribution function where x => 0
    ####
    #### where c is some positive number. We point out that the solution to this
    #### problem is given in Theorem 3.3 of
    #### "Optimal Inequalities in Probability Theory: a Convex Optimization Approach"
    #### by D. Bertsimas and I. Popescu.
    ####
    ################################################################################
    from scipy.stats import expon
    import pandas as pd
    import numpy as np

    c = expon.ppf(0.9)
    H = lambda x: np.array([float(c <= x)])
    mu = 1
    sigma = 2
    limsup = 0

    output = compute_bound(H,mu, sigma, limsup)

    # Check that optimal upper bound is equal to  the analytical solution
    CMsquare = (sigma- mu**2)/mu**2
    delta =  c/mu-1
    pd.DataFrame({'Algorithm': output['bound'], 'Analytical': CMsquare/(CMsquare + delta**2)})

    # Check that the output is feasible
    pd.DataFrame({'moments': [sum(output['P'].p), sum(output['P'].p*output['P'].x),sum(output['P'].p*np.power(output['P'].x,2))], 'truth': [1,mu,sigma]})

    ##############################################################################
    #### Solving a program alike  to (EC.19)
    ##############################################################################
    ####
    #### max P(X > c)
    #### s.t. sum(p)  = 1
    ####      1 <= sum(px) <= 1
    ####      2 <= sum(px**2) <= 2
    ####      (p,x) is a two point mass distribution function where x => 0
    ####
    #### where c is some positive number. This program is the same as the first one
    #### but formulated with inequalities rather than equalities.
    ####
    ################################################################################
    mu = np.array([1,1])
    sigma = np.array([2,2])
    limsup = 0

    output = compute_bound(H,mu, sigma, limsup)
    type(output['bound'])
    # Check that optimal upper bound is equal to  the analytical solution
    pd.DataFrame({'Algorithm': output['bound'], 'Analytical': CMsquare/(CMsquare + delta**2)})

    # Check that the output is feasible
    pd.DataFrame({'moments': [sum(output['P'].p), sum(output['P'].p*output['P'].x),sum(output['P'].p*np.power(output['P'].x,2))], 'truth': [1,mu,sigma]})

    ##############################################################################
    #### Solving a program alike  to (1)
    ##############################################################################
    ####
    #### max P(x >b)
    #### s.t. f(a) = eta
    ####      1-F(a) = beta
    ####      f'(a) => -nu
    ####      f(x) is convex for all x => a
    ####      f(x) is non-negative for all x => a
    ####
    #### where b is some real number larger than a. This problem is of the form of
    #### program (1). By Theorem 4, it is equivalent to solve
    ####
    #### max sum(p H(x))
    #### s.t sum(px) = mu
    ####     sum(px^2) = sigma
    ####     (p,x) is a two point mass distribution function where x => 0
    ####
    #### where mu = eta/nu, sigma = beta/nu, H(x) = 1/2(x + a -b)I(x =a =>b)
    ####
    ################################################################################

    # Assume the true distribution function is a standard exponential
    from scipy.stats import expon
    from robust_tail.optimization_functions import compute_bound
    a = expon.ppf(0.7)
    eta = expon.pdf(a)
    nu = expon.pdf(a)
    beta = 1-expon.cdf(a)
    mu = eta/nu
    sigma = 2*beta/nu
    limsup = 1/2

    # Compute the optimal upper bound for various valus of b
    B = expon.ppf(np.arange(0.7,0.99,0.01))

    bound = []
    for b in B:
        Z = compute_bound(lambda x: 1.0/2*(x + a - b)**2*( x +a >= b), mu, sigma, limsup, nu)['bound'][0]
        bound.append(Z)

    bound = np.array(bound)
    import matplotlib.pyplot as plt
    plt.plot(B,bound)
    plt.ylim(0, 0.3)
    plt.ylabel("Optimal Upper Bound")
    """
    if direction not in direction_type:
        raise ValueError("Invalid direction. Expected one of: %s" % direction_type)

    if not np.isscalar(mu) and not isinstance(mu, np.ndarray):
        raise ValueError("Invalid type for mu. Expected one a scalar or numpy.ndarray.")

    if not np.isscalar(sigma) and not isinstance(sigma, np.ndarray):
        raise ValueError("Invalid type for sigma. Expected one a scalar or numpy.ndarray.")

    n_mu = (1 if np.isscalar(mu) else mu.__len__())
    n_sigma = (1 if np.isscalar(sigma) else sigma.__len__())

    # Check parameters
    if n_mu > 2 or n_mu < 1:
        return "mu must be either a scalar or a vector of length 2."

    if n_sigma > 2 or n_sigma < 1:
        return "sigma must be either a  scalar or vector of length 2."

    # Check that nu >= 0
    scale = np.where(direction == "min", 1, -1)
    if nu < 0:
        return {'bound': scale * float('inf'), 'p': None}

    # Compute bound
    if n_mu == 1 and n_sigma == 1:
        output = _compute_bound_val(H, mu, sigma, limsup, direction)

    if n_mu == 2 and n_sigma == 2:
        output = _compute_bound_int(H, mu, sigma, limsup, direction)

    if n_mu == 1 and n_sigma == 2:
        output = _compute_bound_int(H, np.repeat(mu, 2), sigma, limsup, direction)

    if n_mu == 2 and n_sigma == 1:
        output = _compute_bound_int(H, mu, np.repeat(sigma, 2), limsup, direction)

    output['bound'] = nu*output['bound']
    return output
