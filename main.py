#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 14:10:05 2020

@author: jamilgafur
"""
import genetic_algorithm as GA
import GSA
import numpy as np

np.random.seed(1)
np.set_printoptions(linewidth=np.inf) 


dimension = 5
k = 100
debug = False
# global variables
A, b = GA.cqo_gen_input(k, dimension, debug)
    

# optimization function    
def evalutate_quad_opt(individual):
  x = np.array(individual, dtype=float)
  value = 0.5 * np.matmul(np.matmul(x.T, A), x) - np.matmul(b.T, x)
  return 1/value


def init_gsa(pop, dime, miter, cost_func):
    gsa = GSA.GSA(pop_size=pop, dimension=dime, max_iter=miter)
    gsa.cost_func = cost_func
    return gsa

def init_ga():
    # something here
    return None;

def main():
    # loop over some things and do some kinda study
    gsa = init_gsa(3, 5, 100, evalutate_quad_opt)
    gsa.start()
        


main()