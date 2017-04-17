from robust_tail.W import W
from robust_tail.get_distribution import get_distribution
from robust_tail.optimization_functions import compute_bound

# Import the methods I want to be available at the package level. If I type
#
# from robust_tail import *
#
# then W, get_distribution, compute_bound are imported
#
# If I type,
#
# import robust_tail
#
# then I can access the previous methods as follow
#
# robust_tail.W
# robust_tail.get_distribution
# robust_tail.compute_bound