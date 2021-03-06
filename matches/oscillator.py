import numpy as np
from scipy.integrate import odeint

__all__ = ["mass_spring_deriv", "mass_spring", "simulate_data"]

def mass_spring_deriv(yvec, time, nu_effective, k_effective):
    """
    Damped mass spring oscillator derivatives (y', y'')

    Spring parameters:
    - nu_effective = viscous_coefficient / mass
    - k_effective = spring_constant / mass

    y'' = -nu_effective * y' - k_effective * x


    """
    return (yvec[1], -nu_effective * yvec[1] - k_effective * yvec[0])


def mass_spring(time, nu_effective, k_effective, initial_conditions=(1.0, 0.0)):
    """
    Solves the damped mass spring oscillator equation for the position and first
    derivative
    """
    return odeint(
        mass_spring_deriv, initial_conditions, time,
        args=(nu_effective, k_effective)
    )


def simulate_data(time, nu_effective=None, k_effective=None, initial_conditions=(1.0, 0.0), nsamples=1000, seeds=None):

    ntimesteps = len(time)

    if seeds is not None:
        if len(seeds) != 2:
            raise Exception("seeds must be length 2 (seed for nu, seed for kappa)")

    # create random nu, kappa if the are not supplied
    if nu_effective is None:
        if seeds is not None:
            np.random.seed(seeds[0])
        nu_effective = np.random.rand(nsamples)

    if k_effective is None:
        if seeds is not None:
            np.random.seed(seeds[1])
        k_effective = np.random.rand(nsamples)

    if len(initial_conditions) == 2:
        initial_conditions = np.repeat(np.atleast_2d(initial_conditions), nsamples, axis=0)
    if initial_conditions.shape != (nsamples, 2):
        raise Exception(
            f"initial condition must be shape ({nsamples}, 2). "
            f"The input shape is ({initial_conditions.shape})"
        )

    # initialize data storage for position and derivative
    y = np.empty((nsamples, ntimesteps))
    y_prime = np.empty((nsamples, ntimesteps))

    # loop over samples
    for i in range(nsamples):
        dat = mass_spring(
            time, nu_effective[i], k_effective[i], initial_conditions[i, :]
        )
        y[i, :], y_prime[i, :] = dat[:, 0], dat[:, 1]

    data_dict = {
        'nu_effective': nu_effective,
        'k_effective': k_effective,
        'initial_conditions': initial_conditions,
        'time': time,
        'y': y,
        'y_prime': y_prime
    }

    return data_dict


