# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 15:52:40 2020

@author: Cory
"""

from argparse import ArgumentParser

import random
import numpy.matlib 
import numpy as np 

#For parallel computing
import multiprocessing
import concurrent.futures

from itertools import product

from genetic_algorithm_v2 import OptimizationGA

#For graphing
import matplotlib.pyplot as plt

def print_progress_bar(iteration, total, best_fitness, decimals=3, length=100, fill='#', prefix='Progress:', suffix='Complete--Best fitness: ', printEnd='\r'):
  """
  Variables and progress bar code from:
  https://stackoverflow.com/a/34325723
  """
  percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
  best_fitness_str = ("{0:." + str(decimals) + "f}").format(best_fitness)
  filledLength = int(length * iteration // total)
  bar = fill * filledLength + '-' * (length - filledLength)
  print(f'\r{prefix} |{bar}| {percent}% {suffix} {best_fitness_str}', end = printEnd)
  # Print New Line on Complete
  if iteration == total: 
    print()
    

def _thread_caller(pool, params):
  ga = OptimizationGA(problem=1, pool=pool, debug=-1, pop_size=50, **params)
  fitness, _ =  ga.run()
  return fitness

def grid_search(is_threaded, k, size, pool):
  """
  Performs grid search over all hyperparameters for a particular problem. The
  best values for each hyperparam will be printet when finished to be set as default.

  Parameters
  ----------
  k : Integer
    DESCRIPTION
  size : Integer
    DESCRIPTION.

  Returns
  -------
  None.

  """
  
  
  mu = [1.0, 0.5, 0.0, -0.5, -1.0]
  sigma = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005]
  alpha = [0.1, 0.3, 0.5, 0.7, 0.9]
  indpb = [0.1, 0.3, 0.5, 0.7, 0.9]
  tournsize = [2, 3, 4, 5]
  cxpb = [0.1, 0.3, 0.5, 0.7, 0.9]
  mutpb = [0.1, 0.3, 0.5, 0.7, 0.9]
  total = len(mu) * len(sigma) * len(alpha) * len(indpb) * len(tournsize) * len(cxpb) * len(mutpb)
  iteration = 0
  options = product(mu, sigma, alpha, indpb, tournsize, cxpb, mutpb)
    
  best_values = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  best_fitness = 90000000
  
  if is_threaded:
    executor = multiprocessing.Pool(5)
    submitted = 1
    futures = dict()
  
  try:
    if is_threaded:
      for x in options:
        params = {'size': size, 'k':k, 'mu':x[0], 'sigma':x[1], 'alpha':x[2], 'indpb':x[3], 'tournsize':x[4], 'cxpb':x[5], 'mutpb':x[6]}
        futures[executor.apply_async(_thread_caller, (pool, params))] = x
        print(f'\rSubmitted {submitted} / {total} jobs', end = '\r')
        submitted += 1
    else:
      for params in options:
        ga = OptimizationGA(problem=1, pool=pool, mu=params[0], sigma=params[1], alpha=params[2], indpb=params[3], tournsize=params[4], cxpb=params[5], mutpb=params[6], debug=-1, size=size, k=k, pop_size=50)
        fitness, _ = ga.run()
        fitness = fitness.fitness.values[0]
        
        if fitness <= best_fitness:
          best_fitness = fitness
          best_values = params
          
        print_progress_bar(iteration, total, best_fitness)
        iteration += 1
                    
    if is_threaded:
      for future in futures:
        parameters = futures[future]
        try:
          fitness = future.get()
        except Exception as exc:
          print('%r generated an exception: %s' % (parameters, exc))
          raise exc
        else:
          fitness = fitness.fitness.values[0]
          if fitness <= best_fitness:
            best_fitness = fitness
            best_values = parameters
            
          print_progress_bar(iteration, total, best_fitness)
          iteration += 1
  finally:
    if is_threaded:
      executor.close()
      executor.terminate()
                
              
  print("Best values from grid search evaluation is:\n\tMu:%.3f\n\tSigma:%.3f\n\tAlpha:%.3f\n\tIndpb:%.3f\n\tTournsize:%i\n\tCxpb:%.3f\n\tMutpb:%.3f"
        % best_values)
  print("Best parameters had fitness: %.3f" % (best_fitness))
  
  
def build_parser():
  """
  Builds the parser based on input variables from the command line.
  """
  parser = ArgumentParser()
  
  #Arguments for the Genetic Algorithm
  #=====================================================================================
  parser.add_argument('-p', '--population-size', dest='pop_size', type=int, default=300,
                      help='Number of individuals in the population', 
                      metavar='P')
  
  parser.add_argument('-a', '--alpha', dest='alpha', type=float, default=0.3,
                      help='Alpha used for blending individuals when mating', 
                      metavar='A')
  
  parser.add_argument('-t', '--tournament-size', dest='tournsize', type=int, default=2,
                      help='The number of individuals per tournament during selection', 
                      metavar='T')
  
  parser.add_argument('-i', '--indpb', dest='indpb', type=float, default=0.3,
                      help='Probability that a gene will be mutated in a mutating individual', 
                      metavar='I')
  
  parser.add_argument('-g', '--num-gen', dest='number_generations', type=int, default=1000,
                      help='Number of generations to run through in algorithm', 
                      metavar='G')
  
  parser.add_argument('-c', '--cxpb', dest='cxpb', type=float, default=0.7,
                      help='Percentage chance that 2 individuals will be mated', 
                      metavar='C')
  
  parser.add_argument('-m', '--mutpb', dest='mutpb', type=float, default=0.9,
                      help='Percentage chance that an individual will be mutated', 
                      metavar='M')
  
  parser.add_argument('-mu', '--gausian-mean', dest='mu', type=float, default=0,
                      help='The mean to use in the gausian distrobution when mutating genes', 
                      metavar='MU')
  
  parser.add_argument('-si', '--gausian-sigma', dest='sigma', type=float, default=0.005,
                      help='The standard deviation to use in the gausian distrobution when mutating genes', 
                      metavar='SI')
  #=====================================================================================
  
  #General arguments for problems
  #=====================================================================================
  parser.add_argument('-n', '--size', dest='size', type=int, default=5,
                      help='The size of the square matrices', 
                      metavar='N')
  #=====================================================================================
  
  #Arguments for Quadratic Optimization problem
  #=====================================================================================
  parser.add_argument('-k', '--condition-number', dest='k', type=int, default=3,
                      help='The condition number that we want to approximate for A matrix', 
                      metavar='K')
  #=====================================================================================
  
  #Output arguments
  #=====================================================================================
  parser.add_argument('-v', '--verbose', dest='debug', type=int, default=0,
                      help='The log level for the algorithm. Values are [0, 1, 2]', 
                      metavar='S')
  
  #(TODO)
  #1. have option to output to excel/csv file
  #2. have option to output fitness graph for genetic algorithm run
  #=====================================================================================
  
  #Misc arguments
  #=====================================================================================
  parser.add_argument('-s', '--seed', dest='seed', type=int, default=1234,
                      help='The random seed for the algorithm', 
                      metavar='S')
  
  parser.add_argument('-r', '--problem-runs', dest='problems', type=int, default=2,
                      help='Which problems should be run. 0=just quad, 1=just non-convex, 2=both', 
                      metavar='R')
  
  parser.add_argument('-ai', '--all-inputs', dest='use_pred_inputs', action='store_true',
                      help='If given, all problem inputs will be ignored and each chosen problem will go over all preset inputs')
  
  parser.add_argument('-gs', '--grid-search', dest='perform_grid_search', action='store_true',
                      help='If given, grid search will be performed and then the program will exit.')
  
  parser.add_argument('--distributed', dest='is_distributed', action='store_true',
                      help='If given, a process pool will be made and used to distribut fitness evaluations')
  
  parser.add_argument('--threaded', dest='is_threaded', action='store_true',
                      help='If given, a thread pool will be generated, and multiple GA runs will be done in parallel. Evaluations from all GAs will be sent to same process pool if distributed as well.')
  #=====================================================================================
  
  return parser

def run(options, pool):
  
  if options.problems == 2:
    ga = OptimizationGA(problem=1, pool=pool, **vars(options))
    _, logbook =  ga.run()
  elif options.problems == 0:
    ga = OptimizationGA(problem=1, pool=pool, **vars(options))
    _, logbook =  ga.run()
    
  return logbook

def plot_multi_data(gen, data_dict):
  for key, values in data_dict.items():
    plt.plot(gen, values, 'b-', label=key)
    
  legend = plt.legend(ncol=3, title='Key:', bbox_to_anchor=(1.04, 1))
  plt.xlabel('Generation')
  plt.ylabel("Fitness")
  plt.savefig('ga_' + str(len(gen)) + '.svg', bbox_extra_artists=(legend,), bbox_inches='tight')
  
def plot_single_data(gen, data, key):
  plt.plot(gen, data, 'b-', label=key)
  legend = plt.legend(title='Key:', bbox_to_anchor=(1.04, 1))
  plt.xlabel('Generation')
  plt.ylabel("Fitness")
  plt.savefig('ga__single_' + str(len(gen)) + '.svg', bbox_extra_artists=(legend,), bbox_inches='tight')
  
def main():
  parser = build_parser()
  options = parser.parse_args()
  random.seed(options.seed)
  np.random.seed(options.seed)
  
  if options.is_distributed:
    #For parallel computing
    #multiprocessing.freeze_support()
    pool = multiprocessing.Pool(5)
  else:
    pool = None
  
  try:
    if options.perform_grid_search:
      #Perform Grid search to find best hyperparameters to set as default
      grid_search(options.is_threaded, options.k, options.size, pool)
    else:
      if options.use_pred_inputs:
        for steps in [100, 1000, 10000, 100000]:
          options.number_generations = steps
          log_dict = dict()
          if options.is_threaded:
            executor = concurrent.futures.ThreadPoolExecutor()
            futures = dict()
            
          try:
            for k in [3, 10, 30, 100, 300, 1000]:
              for n in [2, 5, 10, 20, 50, 100]:
                #print("Running for k =  %i, n = %i, steps = %i" % (k, n, steps))
                options.k = k
                options.size = n
                
                if options.is_threaded:
                  futures[executor.submit(run, options, pool)] = "k=" + str(k) + ", n=" + str(n)
                else:
                  logbook = run(options, pool)
                  key = "k=" + str(k) + ", n=" + str(n)
                  log_dict[key] = logbook.select("max")
                  gen = logbook.select('gen')
                
            if options.is_threaded:
              for future in concurrent.futures.as_completed(futures):
                key = futures[future]
                try:
                  logbook = future.result()
                except Exception as exc:
                  print('%r generated an exception: %s' % (key, exc))
                else:
                  log_dict[key] = logbook.select("max")
                  gen = logbook.select('gen')
          finally:
            if options.is_threaded:
              executor.shutdown(wait=False)
              
          plot_multi_data(gen, log_dict)
      else:
        logbook = run(options, pool)
        gen, max_results = logbook.select("gen", "max")
        plot_single_data(gen, max_results, "k=" + str(options.k) + ", n=" + str(options.size))
  finally:
    if options.is_distributed: 
      pool.close()
      pool.terminate()
  
  

if __name__ == '__main__':
  main()