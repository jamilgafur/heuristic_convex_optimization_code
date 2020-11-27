# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 13:20:35 2020

This file is applying Genetic Algorithms to 
Convex quadratic optimization (or linear regression).

@author: Cory Kromer-Edwards
"""

import numpy.matlib 
import numpy as np 
import math
#import random

#random.seed(1234)
#np.random.seed(1234)

#Setting random.seed or np.random.seed sets the seed globally
#which would affect all files that import these while running.
#To avoid this, we create random states for each file that are then
#used for randomness maintaining between generating data for problems
#and running the alogorithms themselves.
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
  
  D = np.matlib.identity(size, dtype = float)
  
  #Since D is initially an identity matrix, we can multiply each row by
  #the d_{i,i} equation and it will only affect d_{i,i}
  def calc_diagonal_entries(x):
    ui = rng.uniform(0.0, 1.0)
    
    #exp((ln k) * ui)
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
  
  Z = rng.normal(0, 1, (size, size))      #Random size x size matrix
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
  
  #Get the condition number calculated from A to check against input k
  #The 3 options are the same out to about 13 decimal places
  #||A||_2||A^-1||_2
  
  #Option 1 to calc conditional number
  #k_check = np.linalg.norm(A, 2) * np.linalg.norm(np.linalg.inv(A), 2)
  
  #Option 2 to calc conditional number
  #eigs = np.linalg.eigvals(A)
  #k_check = np.max(eigs)/np.min(eigs)
  
  #Option 3 to calc conditional number
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
  b = rng.normal(0, 1, (size, 1)) #Generate normally distributed vector (mean=0, std. dev.=1)
  return (A, b)
  

def nonconvex_generate_input(m , M, B, D):
    alpha = np.random.uniform(0, M, size=D)
    beta  = np.random.uniform(1, B, size=D)
    sigma = np.random.uniform(0,2*np.pi, size=D)
    
    return alpha, beta, sigma


if __name__ == '__main__':
  
  #Testing A generator method
  generate_A(3.0, 5)
  
  
  
  