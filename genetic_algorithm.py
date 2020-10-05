# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 09:07:53 2020

@author: Cory Kromer-Edwards
"""
from argparse import ArgumentParser

import random
import numpy.matlib 
import numpy as np 

from deap import base
from deap import creator
from deap import tools

from convex_quadratic_opt import generate_input as cqo_gen_input

#=====================================================================================
#tools: A set of functions to be called during genetic algorithm operations (mutate, mate, select, etc)
#Documentation: https://deap.readthedocs.io/en/master/api/tools.html
#=====================================================================================

#=====================================================================================
#Creator: Creates a class based off of a given class (known as containers)
#   creator.create(class name, base class to inherit from, args**)
#     class name: what the name of the class should be
#     base class: What the created class should inherit from
#     args**: key-value argument pairs that the class should have as fields
#
#   EX: creator.create("FitnessMax", base.Fitness, weights=(1.0,))
#     Creates a class named "FitnessMax" that inherits from base fitness class
#     that library has (maximizes fitness value). It then has a tuple of weights
#     that are given as a field for the class to use later.
#=====================================================================================

#The base.Fitness function will try to maximize fitness*weight, so we want
#negative weight here so the closer we get to 0 (with domain (-inf, 0]) the 
#larger the fitness will become.
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin, generation=0)


#=====================================================================================
#Toolbox: Used to add aliases and fixed arguements for functions that we will use later.
#   toolbox.[un]register(alias name, function, args*)
#     alias name: name to give the function being added
#     function: the function that is being aliased in the toolbox
#     args*: arguments to fix for the function when calling it later
#
#   EX: toolbox.register("attr_bool", random.randint, 0, 1)
#     Creates an alias for the random.randint function with the name "attr_bool"
#       with the default min and max int values being passed in being 0 and 1.
#
#   EX: toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 100)
#     Creates an alias for the tools.initRepeat function with the name "individual".
#     This function takes in the class that we want to repeadidly intialize from, the function to 
#     initialize values with, and how many values to create from that function. This will create
#     an individual with 100 random boolean values.
#=====================================================================================

toolbox = base.Toolbox()

def init_ga_functions(size, alpha, indpb, tournsize):
  """
  Create the function aliases needed to generate the population.

  Parameters
  ----------
  size : Integer
    The size of the x vector.
  alpha : Float
    The blend percentage between both individuals.
  indpb : Float
    Probability to mutate a "gene" (element in x vector).
  tournsize : Integer
    Number of individuals to test against in a single tournament during selection.

  Returns
  -------
  None.

  """
  
  toolbox.register("attr_x", np.random.normal, 0, 1)
  toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_x, size)
  toolbox.register("population", tools.initRepeat, list, toolbox.individual)
  
  toolbox.register("mate", tools.cxBlend, alpha=alpha)
  toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=indpb)
  toolbox.register("select", tools.selTournament, tournsize=tournsize)
  
#Fitness evaluation methods
#=====================================================================================
def evalutate_quad_opt(individual, A=None, b=None):
  x = np.array([individual]).T  #x as Column vector
  return 0.5 * np.matmul(np.matmul(x.T, A), x) - np.matmul(b.T, x)
#=====================================================================================


#Initialization methods
#=====================================================================================
def init_quad_opt(k, size, debug=0):
  A, b = cqo_gen_input(k, size, debug)
  toolbox.register("cqo_evaluate", evalutate_quad_opt, A=A, b=b)
#=====================================================================================

def grid_search(k, size, alpha_values, indpb_values, tournsize_values, cxpb_values, mutpb_values):
  """
  Performs grid search over all hyperparameters for a particular problem. The
  best values for each hyperparam will be printet when finished to be set as default.

  Parameters
  ----------
  size : TYPE
    DESCRIPTION.
  alpha_values : TYPE
    DESCRIPTION.
  indpb_values : TYPE
    DESCRIPTION.
  tournsize_values : TYPE
    DESCRIPTION.
  cxpb_values : TYPE
    DESCRIPTION.
  mutpb_values : TYPE
    DESCRIPTION.

  Returns
  -------
  None.

  """
  
  total_runs = len(alpha_values) * len(indpb_values) * len(tournsize_values) * len(cxpb_values) * len(mutpb_values)
  
  def unregister_funcs():
    toolbox.unregister("attr_x")
    toolbox.unregister("individual")
    toolbox.unregister("population")
    
    toolbox.unregister("mate")
    toolbox.unregister("mutate")
    toolbox.unregister("select")
    
  best_values = (0.0, 0.0, 0.0, 0.0, 0.0)
  best_fitness = -1000000
  init_quad_opt(k, size)
  
  run_num = 0
  
  for a in alpha_values:
    for i in indpb_values:
      for t in tournsize_values:
        for c in cxpb_values:
          for m in mutpb_values:
            init_ga_functions(size, a, i, t)
            fitness = run_ga(toolbox.cqo_evaluate, 300, c, m, debug=-1).fitness.values[0]
            unregister_funcs()
            
            run_perc = run_num / total_runs * 100.0
            if run_perc % 10 <= 0.05:
              print("Gird search at %.3f percent with best fitness of %.3f" % (run_perc, best_fitness))
            
            run_num += 1
            
            if fitness >= best_fitness:
              best_fitness = fitness
              best_values = (a, i, t, c, m)
              
  print("Best values from grid search evaluation is:\n\tAlpha:%.3f\n\tIndpb:%.3f\n\tTournsize:%i\n\tCxpb:%.3f\n\tMutpb:%.3f"
        % best_values)
  print("Best parameters had fitness: %.3f" % (best_fitness))
  
  toolbox.unregister("cqo_evaluate")
  

def run_ga(eval_function, num_gen, cxpb, mutpb, debug=0):
  """
  Run a genetic algorithm with the given evaluation function and input parameters.
  Main portion of code for this method found from Deap example at URL:
  https://deap.readthedocs.io/en/master/overview.html
  
  (TODO)
  Will build statics and plot outputs before returning

  Parameters
  ----------
  size : Integer
    The size of the x vector.
  eval_function : function
    The evaluation to use for a particular problem.
  num_gen : Integer
    Number of generations to run over.
  cxpb : Float
    Percentage chance that 2 individuals will be mated.
  mutpb : Float
    Percentage chance that an individual will be mutated.
  alpha : Float
    The blend percentage between both individuals.
  indpb : Float
    Probability to mutate a "gene" (element in x vector).
  tournsize : Integer
    Number of individuals to test against in a single tournament during selection.
  debug : Integer, optional
    Debug level.

  Returns
  -------
  Best individual out of all generations.

  """
  pop = toolbox.population(n=50)
  hof = tools.HallOfFame(25)
  
  # Evaluate the entire population
  fitnesses = list(map(eval_function, pop))
  for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

  for g in range(num_gen):
    # Select the next generation individuals (with replacement)
    offspring = toolbox.select(pop, len(pop))
    # Clone the selected individuals (since selection only took references rather than values)
    offspring = list(map(toolbox.clone, offspring))
    
    # Apply crossover and mutation on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
      if random.random() < cxpb:
        toolbox.mate(child1, child2)
        del child1.fitness.values
        del child2.fitness.values

    for mutant in offspring:
      if random.random() < mutpb:
        toolbox.mutate(mutant)
        del mutant.fitness.values

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(eval_function, invalid_ind)
    
    if debug >= 2:
      print("Generation %i has min fitness value: %.3f" % (g, -max(fit)))
    elif debug == 1 and g % 10 == 0:
      print("Generation %i has min fitness value: %.3f" % (g, -max(fit)))
    
    for ind, fit in zip(invalid_ind, fitnesses):
      ind.fitness.values = fit
      ind.generation = g
    
    # The population is entirely replaced by the offspring
    pop[:] = offspring
    hof.update(pop)
    
  if debug >= 0:
    print("Convex quadratic optimization problem results:")
    print("\tBest individual seen in all generations:\t%r" % (hof[0]))
    print("\tBest individual seen fitness value:\t\t%.3f" % (-hof[0].fitness.values[0]))
    print("\tBest individual seen generation appeared in:\t%i" % (hof[0].generation))
    
  return hof[0]

  
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
  
  parser.add_argument('-a', '--alpha', dest='alpha', type=float, default=0.9,
                      help='Alpha used for blending individuals when mating', 
                      metavar='A')
  
  parser.add_argument('-t', '--tournament-size', dest='tournsize', type=int, default=2,
                      help='The number of individuals per tournament during selection', 
                      metavar='T')
  
  parser.add_argument('-i', '--indpb', dest='indpb', type=float, default=0.9,
                      help='Probability that a gene will be mutated in a mutating individual', 
                      metavar='I')
  
  parser.add_argument('-g', '--num-gen', dest='number_generations', type=int, default=1000,
                      help='Number of generations to run through in algorithm', 
                      metavar='G')
  
  parser.add_argument('-c', '--cxpb', dest='cxpb', type=float, default=0.9,
                      help='Percentage chance that 2 individuals will be mated', 
                      metavar='C')
  
  parser.add_argument('-m', '--mutpb', dest='mutpb', type=float, default=0.7,
                      help='Percentage chance that an individual will be mutated', 
                      metavar='M')
  #=====================================================================================
  
  #General arguments for problems
  #=====================================================================================
  parser.add_argument('-n', '--size', dest='size', type=int, default=5,
                      help='The size of the square matrices', 
                      metavar='N')
  #=====================================================================================
  
  #Arguments for Quadratic Optimization problem
  #=====================================================================================
  parser.add_argument('-k', '--condition-number', dest='k', type=float, default=3,
                      help='The condition number that we want to approximate for A matrix', 
                      metavar='K')
  #=====================================================================================
  
  #Misc arguments
  #=====================================================================================
  parser.add_argument('-s', '--seed', dest='seed', type=int, default=1234,
                      help='The random seed for the algorithm', 
                      metavar='S')
  
  parser.add_argument('-v', '--verbose', dest='debug', type=int, default=0,
                      help='The log level for the algorithm. Values are [0, 1, 2]', 
                      metavar='S')
  
  parser.add_argument('-r', '--problem-runs', dest='problems', type=int, default=2,
                      help='Which problems should be run. 0=just quad, 1=just non-convex, 2=both', 
                      metavar='R')
  
  parser.add_argument('-ai', '--all-inputs', dest='use_pred_inputs', action='store_true',
                      help='If given, all problem inputs will be ignored and each chosen problem will go over all preset inputs.')
  #=====================================================================================
  
  return parser

def run(options, size, k, number_generations):
  init_ga_functions(size, options.alpha, options.indpb, options.tournsize)
  
  if options.problems == 2:
    init_quad_opt(k, size, options.debug)
    run_ga(toolbox.cqo_evaluate, number_generations, options.cxpb, options.mutpb, options.debug)
  elif options.problems == 0:
    init_quad_opt(k, size, options.debug)
    run_ga(toolbox.cqo_evaluate, number_generations, options.cxpb, options.mutpb, options.debug)
  
def main():
  parser = build_parser()
  options = parser.parse_args()
  random.seed(options.seed)
  np.random.seed(options.seed)
  
  #Perform Grid search to find best hyperparameters to set as default
  #grid_search(options.k, options.size, [0.1, 0.3, 0.5, 0.7, 0.9], [0.1, 0.3, 0.5, 0.7, 0.9],
  #            [2, 3, 4, 5], [0.1, 0.3, 0.5, 0.7, 0.9], [0.1, 0.3, 0.5, 0.7, 0.9])
  
  if options.use_pred_inputs:
    for k in [3, 10, 30, 100, 300, 1000]:
      for n in [2, 5, 10, 20, 50, 100]:
        for steps in [100, 1000, 10000, 100000]:
          print("Running for k =  %i, n = %i, steps = %i" % (k, n, steps))
          run(options, k, n, steps)
          print("")
  else:
    run(options, options.size, options.k, options.number_generations)
  
  

if __name__ == '__main__':
  main()

