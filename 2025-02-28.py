import numpy as np

#import jax
#import jax.numpy as jnp

from scipy import optimize
import timeit


## Exercise 1
def f(x):
    return np.exp(x[0]-1) + np.exp(-x[1]+1) + (x[0]-x[1])**2

def grad(x):
    return [
        np.exp(x[0]-1) + 2*(x[0]-x[1]),
       -np.exp(-x[1]+1) - 2*(x[0]-x[1])
    ]

grad_jax = jax.grad(f)


x0 = [1.,0.]
optimize.minimize(fun = f, x0=x0, jac = grad, method="BFGS")
optimize.minimize(fun = f, x0=x0, jac = grad, method="CG")
optimize.minimize(fun = f, x0=x0, jac = grad, method="Nelder-Mead")
optimize.minimize(fun = f, x0=x0, jac = grad, method="Newton-CG", tol=1e-12)

timeit.Timer(lambda: optimize.minimize(fun = f, x0=x0, jac = grad, method="BFGS")).repeat(1,100)
timeit.Timer(lambda: optimize.minimize(fun = f, x0=x0, jac = grad, method="CG")).repeat(1,100)
timeit.Timer(lambda: optimize.minimize(fun = f, x0=x0, jac = grad, method="Nelder-Mead")).repeat(1,100)

## Exercise 2

from scipy.stats import gamma

g = gamma(a=2., scale=2.)
x = g.rvs(size=100, random_state=1234)

def mle_gamma(theta):
    if theta[0] < 0 or theta[1] < 0:
        return np.inf

    return -np.sum(gamma.logpdf(x, a=theta[0], scale=theta[1]))

mle_gamma([1.,1.])

optimize.minimize(
    mle_gamma, x0=[1.,1.], method="bfgs"
)

gamma.fit(x, floc=0)
