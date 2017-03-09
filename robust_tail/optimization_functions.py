# git clone https://github.com/sgubianpm/pygensa.git
#import pip
#pip.main(["install","--user", "--upgrade","/Users/cmottet/Packages/python/pygensa"])

import numpy as np
import pandas as pd
from pygensa.gensa import gensa
from .W import W
from .get_distribution import get_distribution


direction_type = ['max', 'min']


def _compute_bound_val(H,  mu, sigma, limsup, direction='max'):
    """
    Solves programs (5) over 2-point masses distribution functions, when nu = 1.

    :param H:
    :param mu:
    :param sigma:
    :param limsup:
    :param direction:
    :return:

    :Example:

    1+1

    """
    W(1)
    if direction not in direction_type:
        raise ValueError("Invalid direction. Expected one of: %s" % direction_type)

    scale = np.where(direction == 'min', 1, -1)

    # Exclusion of trivial scenarios
    if mu**2 > sigma:
        output = {'bound': scale*float('inf'), 'P': None}

    if mu**2 == sigma:
        output = {'bound': H(mu), 'P': get_distribution(mu, sigma)}

    # Treatment of non-trivial scenarios
    if mu**2 < sigma:
        z = gensa(func=lambda x1: scale*W(x1, mu, sigma, limsup, H), x0=None, bounds=[[0, mu]])
        output = {'bound': scale*z.fun, 'P': get_distribution(mu, sigma, x1=z.x)}

    return output


###
### Solves programs (EC.19) over 2-point masses distribution functions, when nu = 1.
###
# computeBoundInt <- function(H,mu, sigma, lambda, direction = c("max", "min"))
# {
#   direction <- match.arg(direction)
#   scale <- if (direction == "min") 1 else -1
#
#   # Exclude trival scenarios
#   if ( (mu[1] > mu[2]) || (sigma[1] > sigma[2])) output <- list(bound = scale*Inf, P = NULL)
#   if (mu[1]^2 > sigma[2])  output <- list(bound = scale*Inf, P = NULL)
#   if (mu[1]^2 == sigma[2]) output <- list(bound = H(mu), P = getDistribution(mu[1],sigma[2]))
#
#   # Non-trivial scenario
#   if (mu[1]^2 < sigma[2])
#   {
#     # First subprogram
#     lower <- c(0,max(sigma[1],mu[2]^2))
#     upper <- c(mu[2],sigma[2])
#
#     if (any(lower > upper)) { Z1 <- list(value = Inf) ; P1 <- list(NULL)
#     } else{
#       Z1 <- GenSAmodified(fn = function(args) scale*W(x = args[1], mu[2], rho = args[2], lambda, H),
#                           par         = lower,
#                           lower       = lower,
#                           upper       = upper)
#
#       P1 <- getDistribution(mu = mu[2],sigma = Z1$par[2],x1 = Z1$par[1])
#     }
#
#
#     # Second Subprogram
#     lower <- c(0, mu[1])
#     upper <- rep(min(mu[2], sqrt(sigma[2])),2)
#
#     if (any(lower > upper)){ Z2 <- list(value = Inf) ; P2 <- list(NULL)
#     } else{
#       f2 <- function(args){ if (args[1] > args[2]) Inf else scale*W(x = args[1], w = args[2], sigma[2], lambda, H)}
#       Z2 <- GenSAmodified(fn = f2,
#                           par   = lower,
#                           lower = lower,
#                           upper = upper)
#       P2 <- getDistribution(mu = Z2$par[2], sigma =  sigma[2],x1 = Z2$par[1])
#     }
#
#     bound <- max(scale*Z1$value,scale*Z2$value)
#     P <- if (bound == scale*Z1$value) P1 else P2
#
#     output <- list(bound = bound, P = P)
#   }
#
#   return(output)
# }
#
# #' Solves programs (5) and (EC.19) over 2-point masses distribution functions
# #'
# #' This function solves programs (5) and (EC.19), in the case
# #' when their respective feasible region is restricted to distribution functions with at most two point supports.
# #'
# #' @param H The function defined in the objective value of program  (5) and (EC.19)
# #' @param mu Either an ordered vector containing bounds of the first moment in program (EC.19), or a scalar
# #' value as in program (5)
# #' @param sigma Either an ordered vector containing bounds of the second moment in program (EC.19), or a scalar
# #' value as in program (5)
# #' @param lambda A real number  giving the limit value of the ratio H(x)/x^2 when x goes to infinity
# #' @param nu A real number as defined in program (5) and  (EC.19) (actually denoted nubar in the latter case)
# #' @param direction A string either \emph{min} or \emph{max} identifying the type of wether program (5)
# #'  should be a min or a max program. Default is \emph{max}.
# #' @return
# #' \item{bound}{the optimal objective value}
# #' \item{P}{the optimal distribution function with point masses p = (p1,p2) and point supports x = (x1,x2)}
# #' @export
# #'
# #' @examples
# #' ##############################################################################
# #' #### Solving a program alike  to (5)
# #' ##############################################################################
# #' ####
# #' #### max P(X > c)
# #' #### s.t. sum(px) = 1
# #' ####      sum(px^2) = 2
# #' ####      (p,x) is a two point mass distribution function where x => 0
# #' ####
# #' #### where c is some positive number. We point out that the solution to this
# #' #### problem is given in Theorem 3.3 of
# #' #### "Optimal Inequalities in Probability Theory: a Convex Optimization Approach"
# #' #### by D. Bertsimas and I. Popescu.
# #' ####
# #' ################################################################################
# #'
# #' c <- qexp(0.9)
# #' H <- function(x) as.numeric(c <= x)
# #' mu <- 1
# #' sigma <- 2
# #' lambda <- 0
# #'
# #' output <- computeBound(H,mu, sigma, lambda)
# #'
# #' # Check that optimal upper bound is equal to  the analytical solution
# #' CMsquare <- (sigma- mu^2)/mu^2
# #' delta <-  c/mu-1
# #' data.frame(Algorithm = output$bound, Analytical = CMsquare/(CMsquare + delta^2))
# #'
# #' # Check that the output is feasible
# #' with(output$P, data.frame(moments = c(sum(p),sum(p*x), sum(p*x^2)), truth = c(1,mu,sigma) ))
# #'
# #' ##############################################################################
# #' #### Solving a program alike  to (EC.19)
# #' ##############################################################################
# #' ####
# #' #### max P(X > c)
# #' #### s.t. sum(p)  = 1
# #' ####      1 <= sum(px) <= 1
# #' ####      2 <= sum(px^2) <= 2
# #' ####      (p,x) is a two point mass distribution function where x => 0
# #' ####
# #' #### where c is some positive number. This program is the same as the first one
# #' #### but formulated with inequalities rather than equalities.
# #' ####
# #' ################################################################################
# #' mu <- c(1,1)
# #' sigma <- c(2,2)
# #' lambda <- 0
# #'
# #' output <- computeBound(H,mu, sigma, lambda)
# #'
# #' # Check that optimal upper bound is equal to  the analytical solution
# #' data.frame(Algorithm = output$bound, Analytical = CMsquare/(CMsquare + delta^2))
# #'
# #' # Check that the output is feasible
# #' with(output$P, data.frame(lowerMomentBound = c(1, mu[1], sigma[1]), moments = c(sum(p),sum(p*x), sum(p*x^2)), upperMomentBound = c(1, mu[2], sigma[2]) ))
# #'
# #' ##############################################################################
# #' #### Solving a program alike  to (1)
# #' ##############################################################################
# #' ####
# #' #### max P(x >b)
# #' #### s.t. f(a) = eta
# #' ####      1-F(a) = beta
# #' ####      f'(a) => -nu
# #' ####      f(x) is convex for all x => a
# #' ####      f(x) is non-negative for all x => a
# #' ####
# #' #### where b is some real number larger than a. This problem is of the form of
# #' #### program (1). By Theorem 4, it is equivalent to solve
# #' ####
# #' #### max sum(p H(x))
# #' #### s.t sum(px) = mu
# #' ####     sum(px^2) = sigma
# #' ####     (p,x) is a two point mass distribution function where x => 0
# #' ####
# #' #### where mu = eta/nu, sigma = beta/nu, H(x) = 1/2(x + a -b)I(x =a =>b)
# #' ####
# #' ################################################################################
# #'
# #' # Assume the true distribution function is a standard exponential
# #' a <- qexp(0.7)
# #' eta <- dexp(a)
# #' nu <- dexp(a)
# #' beta <- 1-pexp(a)
# #' mu <- eta/nu
# #' sigma <- 2*beta/nu
# #' lambda <- 1/2
# #'
# #' # Compute the optimal upper bound for various valus of b
# #' b <- qexp(seq(0.7,0.99, by = 0.01))
# #'
# #' runFunc <- function(b){
# #' H <- function(x) 1/2*(x + a - b)^2*( x +a >= b)
# #' bound <- RobustTail::computeBound(H,mu,sigma,lambda,nu)$bound
# #' output <- data.frame(b = b, bound = bound)
# #' }
# #'
# #' dataPlot <- plyr::ldply(lapply(X = b, FUN = runFunc))
# #'
# #' library(ggplot2)
# #' ggplot(dataPlot, aes(x = b, y = bound)) +
# #' geom_line() +
# #' labs(y = "Optimal Upper Bound") +
# #' ylim(c(0,0.3))
# computeBound = function(H, mu, sigma, lambda, nu = 1, direction = c("max", "min"))
# {
#   direction <- match.arg(direction)
#
#   Nmu <- length(mu)
#   Nsigma <- length(sigma)
#
#   # Check parameters
#   if (Nmu > 2 | Nmu < 1) return("mu must be either a scalar or a vector of length 2.")
#   if (Nsigma > 2 | Nsigma < 1) return("sigma must be either a  scalar or vector of length 2.")
#
#   # Check that nu >= 0
#   scale <- if (direction == "min") 1 else -1
#   if (nu < 0) return(list(bound = scale*Inf, P = NULL))
#
#   # Compute bound
#   if (Nmu == 1 & Nsigma == 1) output <- computeBoundVal(H, mu, sigma, lambda, direction)
#   if (Nmu == 2 & Nsigma == 2) output <- computeBoundInt(H, mu, sigma, lambda, direction)
#   if (Nmu == 1 & Nsigma == 2) output <- computeBoundInt(H, rep(mu,2), sigma, lambda, direction)
#   if (Nmu == 2 & Nsigma == 1) output <- computeBoundInt(H, mu, rep(sigma,2), lambda, direction)
#
#   output$bound <- nu*output$bound
#   return(output)
# }
