from inspect import isfunction
from scipy.optimize import OptimizeResult
from pygensa.gensa import gensa
import numpy as np




def _gensa_modified(func, x0, bounds, maxiter=500, initial_temp=5230., visit=2.62,accept=-5.0, maxfun=1e7, args=(), seed=None, pure_sa=False):
    """Extension of the gensa function available the pygensa package at https://github.com/sgubianpm/pygensa

    This function is an extension of the function gensa defined in the package pygensa at
    https://github.com/sgubianpm/pygensa. The only difference with the existing version and _gensa_modified is that
    it allows the user to pass lower bounds and upper bounds with equal values. Though this is a trivial
    scenario in which case the optimal solution should be lower = upper, the current version of gensa crashes.

    :param fun : callable
        The objective function
    :param x0 : ndarray
        The starting coordinates.
    :param bounds : sequence
        Bounds for variables.  ``(min, max)`` pairs for each element in ``x``,
        defining the lower and upper bounds for the optimizing argument of
        `func`. It is required to have ``len(bounds) == len(x)``.
        ``len(bounds)`` is used to determine the number of parameters in ``x``.
    :param args : tuple, optional
        Any additional fixed parameters needed to
        completely specify the objective function.
    :param seed : int or `np.random.RandomState`, optional
        If `seed` is not specified the `np.RandomState` singleton is used.
        If `seed` is an int, a new `np.random.RandomState` instance is used,
        seeded with seed.
        If `seed` is already a `np.random.RandomState instance`, then that
        `np.random.RandomState` instance is used.
        Specify `seed` for repeatable minimizations. The random numbers
        generated with this seed only affect the visiting distribution
        function and new coordinates generation.
    :param temp_start : float, optional
        The initial temperature, use higher values to facilitates a wider
        search of the energy landscape, allowing gensa to escape local minima
        that it is trapped in.
    :param qv : float, optional
        Parameter for visiting distribution. Higher values give the visiting
        distribution a heavier tail, this makes the algorithm jump to a more
        distant region. The value range is (0, 3]
    :param qa : float, optional
        Parameter for acceptance distribution. It is used to control the
        probability of acceptance. The lower the acceptance parameter, the
        smaller the probability of acceptance. It has to be any negative value.
    :param maxfun : int, optional
        Soft limit for the number of objective function calls. If the
        algorithm is in the middle of a local search, this number will be
        exceeded, the algorithm will stop just after the local search is
        done.
    :param maxsteps: int, optional
        The maximum number of gensa iterations will perform.

    :return:

    :Example:

    from robust_tail.gensa_modified import _gensa_modified
    from pygensa.gensa import gensa

    # Test gensa_modified in the univariate case
    def f1(x): return x
    output = _gensa_modified(func = f1,x0 = None, bounds = [[1,1]])
    output
    gensa(func = f1,x0 = None, bounds = [[1,1]]) # This crashes

    # Test gensa_modified in the bivariate case with one lower bound equal to the upper bound
    def f2(x): return x[0] + x[1]
    output = _gensa_modified(func = f2,x0 = None,bounds = [[1,1],[1,3]])
    output.x
    output.fun

    gensa(func = f2,x0 = None,bounds = [[1,1],[1,3]]) # This crashes

    # Test gensa_modified in the bivariate case with both lower bound are equal to the upper bounds
    output = _gensa_modified(func = f2,x0 = None,bounds = [[1,1],[1,1]])
    output.x
    output.fun

    gensa(func = f2,x0 = None,bounds = [[1,1],[1,1]]) # This crashes

    # Let's check that when the lower bounds are strictly smaller than the upper bounds,
    # all goes well)
    output_gensa_modified = _gensa_modified(func = f2,x0 = None,bounds=[[1,2],[2,3]])
    output_gensa = gensa(func = f2,x0 = None,bounds=[[1,2],[2,3]])

    # Check that they have the same optimal solution
    all(output_gensa_modified.x == output_gensa.x)

     # Check that they have the same optimal objective value
     output_gensa_modified.fun == output_gensa.fun
    """

    # Check necessary conditions to run gensa_modified
    if not isfunction(func) or func is None:
        print("func has to be function.")
        return None

    for bound in bounds:
        if len(bound) != 2:
            print("Each parameter needs a lower and upper bounds")
            return None

    # If all lower bounds are different from the upper bounds,
    # run the usual gensa algorithm
    samebound = [x[0] == x[1] for x in bounds]
    lower_bound = [x[0] for x in bounds]

    if not any(samebound):
        output = gensa(func, x0, bounds, maxiter, initial_temp, visit, accept, maxfun, args, seed, pure_sa)
    else:
        index = np.where(samebound)

        def _new_func(new_x=None):
            if new_x is not None:
                new_x_copy = list(new_x)
                x = np.array([bound[0] if bound[0] == bound[1] else new_x_copy.pop(0) for bound in bounds])
            else:
                x = np.array([bound[0] for bound in bounds])
            output = func(x)
            return output

        if all(samebound):
            output = OptimizeResult()
            output.x = lower_bound
            output.fun = _new_func()
            return output

        new_bounds = [bound for bound in bounds if bound[0] != bound[1]]

        if x0 is not None:
            new_x0 = np.array([x0[i] for i in range(0, len(bounds)) if bounds[i][0] != bounds[i][1]])
        else:
            new_x0 = None

        output = gensa(_new_func, new_x0, new_bounds, maxiter, initial_temp, visit, accept, maxfun, args, seed, pure_sa)

        # The output vector par of gensa will be of the same dimension as the new_lower bound
        # we need to include the value of lower_bound that was discarded in "par"
        if isinstance(OptimizeResult(), OptimizeResult):
            output_x_copy = list(output.x[:])
            output.x = [bound[0] if bound[0] == bound[1] else output_x_copy.pop(0) for bound in bounds]

    return output


