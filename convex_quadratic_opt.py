# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 13:20:35 2020

This file is applying Genetic Algorithms to 
Convex quadratic optimization (or linear regression).

@author: Cory Kromer-Edwards
"""

import matplotlib.pyplot as plt
import numpy as np
import math

# import random

# random.seed(1234)
# np.random.seed(1234)

# Setting random.seed or np.random.seed sets the seed globally
# which would affect all files that import these while running.
# To avoid this, we create random states for each file that are then
# used for randomness maintaining between generating data for problems
# and running the alogorithms themselves.
rng = np.random.RandomState(1234)


def generate_D(k, size):
    """
  Generate matrix D such that:
    with diagonal entries dii = exp((ln κ)ui) 
    with ui uniformly distributed over [0, 1].

  Parameters
  ----------
  k : Float
    Condition number of A.
  size : Integer
    size of square matrix D.

  Returns
  -------
  diagonal matrix D

  """

    D = np.identity(size, dtype=float)

    # Since D is initially an identity matrix, we can multiply each row by
    # the d_{i,i} equation and it will only affect d_{i,i}
    def calc_diagonal_entries(x):
        ui = rng.uniform(0.0, 1.0)

        # exp((ln k) * ui)
        return x * math.exp(math.log(k) * ui)

    D = np.apply_along_axis(calc_diagonal_entries, axis=1, arr=D)
    return D


def generate_Q(size):
    """
  Generate matrix Q such that:
    Z = pseudo-random size x size matrix
    R = upper triangular matrix with nonzero values
    Solve for Q in QR factorization equation: Z = QR

  Parameters
  ----------
  size : Integer
    size of square matrix Q.

  Returns
  -------
  Q matrix

  """

    Z = rng.normal(0, 1, (size, size))  # Random size x size matrix
    Q, R = np.linalg.qr(Z)
    return Q


def generate_A(k, size, debug=0):
    """
  Generates A matrix such that:
    A = (Q^T)DQ
    and
    k_2(A) = ||A||_2||A^-1||_2 is about input k
    where:
      Q is generated from above method
      D is generated from above method

  Parameters
  ----------
  k : Float
    Condition number of A.
  size : Integer
    size of square matrix D.

  Returns
  -------
  A matrix

  """

    Q = generate_Q(size)
    D = generate_D(k, size)
    A = np.matmul(np.matmul(Q.transpose(), D), Q)

    # Get the condition number calculated from A to check against input k
    # The 3 options are the same out to about 13 decimal places
    # ||A||_2||A^-1||_2

    # Option 1 to calc conditional number
    # k_check = np.linalg.norm(A, 2) * np.linalg.norm(np.linalg.inv(A), 2)

    # Option 2 to calc conditional number
    # eigs = np.linalg.eigvals(A)
    # k_check = np.max(eigs)/np.min(eigs)

    # Option 3 to calc conditional number
    k_check = np.linalg.cond(A)

    if debug >= 1:
        print("Input condition number: %r" % (k))
        print("Caclulated (approximate) condition number: %r" % (k_check))

    return A


def generate_input(k, size, debug=0):
    """
  Generate A matrix and b vector

  Parameters
  ----------
  k : Float
    Condition number of A.
  size : Integer
    size of square matrix D.

  Returns
  -------
  (A matrix, b vector)

  """
    A = generate_A(k, size, debug)
    b = rng.normal(0, 1, (size, 1))  # Generate normally distributed vector (mean=0, std. dev.=1)
    solution = generate_solution_convex(A, b)
    return A, b, solution


def generate_solution_convex(A, b):
    return np.matmul(np.linalg.inv(A), b)


def generate_solution_nonconvex(Q, alpha, beta, gamma):
    z = np.linspace(-10, 10, 10000)
    g_j_min = []
    for row in range(Q.shape[0]):
        y = []
        for n in range(len(z)):
            y.append(g_j_test(z[n], alpha[:, row], beta[:, row], \
                              gamma[:, row]))
        star_loc = y.index(min(y))
        g_j_min.append(z[star_loc])

    g_j_min = np.transpose(Q) @ g_j_min

    return g_j_min


# Problem 2: Highly non-convex optimization
# ===========================================================================================================
def nonconvex_generate_input(size, m, M, b):
    # Generate Q
    Q = generate_Q(size)

    alpha = np.random.uniform(0, M, size=(m, size))
    beta = np.random.uniform(1, b ** 2, size=(m, size))
    gamma = np.random.uniform(0, 2 * np.pi, size=(m, size))
    return Q, alpha, beta, gamma


def preview_nonconv(size, m, M, b):
    z = np.linspace(-10, 10, 10000)
    Q = generate_Q(size)
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure()
    ax = plt.axes()
    g_j_minimums = []
    alphas = []
    betas = []
    gammas = []
    for j in range(size):
        alpha = np.random.uniform(0, M, size=(m, 1))
        alphas.append(alpha)
        beta = np.random.uniform(1, b ** 2, size=(m, 1))
        betas.append(beta)
        gamma = np.random.uniform(0, 2 * np.pi, size=(m, 1))
        gammas.append(gamma)
        y = []
        for n in range(len(z)):
            y.append(g_j_test(z[n], alpha, beta, gamma))

        ax.plot(z, y)
        g_j_minimums.append(g_j_test(0, alpha, beta, gamma))

    print(f"g_j global minimizers: {g_j_minimums}")
    print(f"f(x) global minimizer: {sum(g_j_minimums)}")
    glob_z = np.array([0 for _ in range(size)])
    print(f"Minimum X: {np.matmul(Q.T, glob_z)}")

    # Testing vectorized functions
    alphas = np.array(alphas).reshape((size, m)).T
    betas = np.array(betas).reshape((size, m)).T
    gammas = np.array(gammas).reshape((size, m)).T
    new_z = np.array([[0 for _ in range(size)]]).T
    print(f"new Z shape: {new_z.shape}")
    print(f"Shape of Alpha: {alphas.shape}")
    g_vect = g_j_vect_test(new_z, alphas, betas, gammas)
    print(g_vect)
    print(f"Vectorized f(x) (using g_j_vect) minimum: {np.sum(g_vect)}")
    print(f"Vectorized f(x) (using f_vect) minimum: {f_vect_test(new_z, alphas, betas, gammas)}")
    plt.show()


def g_j_test(z, alpha, beta, gamma):
    """
    Unvectorized g_j()
    :param z: Integer value
    :param alpha: ndarray with shape (m, 1)
    :param beta: ndarray with shape (m, 1)
    :param gamma: ndarray with shape (m, 1)
    :return: g_j(z_j)
    """
    return (0.5 * (z ** 2)) + np.sum(alpha * np.cos((beta * z) + gamma))


def g_j_vect_test(z, alpha, beta, gamma):
    """
    Vectorized g_j()
    :param z: ndarray with shape (n, 1)
    :param alpha: ndarray with shape (m, n)
    :param beta: ndarray with shape (m, n)
    :param gamma: ndarray with shape (m, n)
    :return: all g_j(z_j) for j from 1 to n
    """
    return (0.5 * np.matmul(z.T, z)) + np.sum(np.multiply(alpha, np.cos((np.matmul(beta, z) + gamma))), axis=0)


def f_vect_test(z, alpha, beta, gamma):
    """
    Vectorized f(x) (for testing in preview_nonconv method)
    It is provided with z instead like other test functions as if z is precomputed
    :param z: ndarray with shape (n, 1)
    :param alpha: ndarray with shape (m, n)
    :param beta: ndarray with shape (m, n)
    :param gamma: ndarray with shape (m, n)
    :return: f(x)
    """
    return ((0.5 * np.matmul(z.T, z)) + np.sum(np.multiply(alpha, np.cos((np.matmul(beta, z) + gamma)))))[0, 0]


def f_vect(x, Q, alpha, beta, gamma):
    """
    Problem 2's f(x) function calculation.
    :param x: ndarray with shape (n, 1)
    :param Q: ndarray with shape (n, n)
    :param alpha: ndarray with shape (m, n)
    :param beta: ndarray with shape (m, n)
    :param gamma: ndarray with shape (m, n)
    :return: f(x)
    """
    z = np.matmul(Q, x)
    return np.dot(z, z) / 2 + np.sum(alpha * np.cos(np.array(z, ndmin=2) * beta + gamma), axis=(0, 1))


# ===========================================================================================================

if __name__ == '__main__':
    # Testing A generator method
    # generate_A(3.0, 5)
    preview_nonconv(5, 3, 5, 3)
