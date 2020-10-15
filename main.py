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


A , b = None, None
# optimization function    
def evalutate_quad_opt(individual):
  x = np.array(individual, dtype=float)
  value = 0.5 * np.matmul(np.matmul(x.T, A), x) - np.matmul(b.T, x)
  return 1/value


def init_gsa(pop, dime, miter, cost_func):
    gsa = GSA.GSA(pop_size=pop, dimension=dime, max_iter=miter)
    gsa.cost_func = cost_func
    return gsa


def main():

    global A
    global b
    # we want to study rate of convergence for population and iteration size 
    # at different dimensions
    # we define converge after 10 iteration amount change <= .0007
    for dimension in range(2,10,2):
        A, b= GA.cqo_gen_input(100, dimension, False)
        for pop in range(10,100,10):
            gsa = init_gsa(pop, dimension, 100, evalutate_quad_opt)
            gsa_output = gsa.start()    
            text_file = open("GSA_dim_{}_pop{}_iter{}.txt".format(dimension, pop, 100), "w")
            text_file.write(gsa_output)
            text_file.close()
        


main()