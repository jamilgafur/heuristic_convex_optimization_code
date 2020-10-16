# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 15:55:47 2020

@author: Cory
"""

import random
import numpy.matlib 
import numpy as np 

from deap import base
from deap import creator
from deap import tools

#Imports for problems
from convex_quadratic_opt import generate_input as cqo_gen_input

#Set up for numpy warnings within the fitness evaluation methods
#By default, warnings are just printed to stderr rather than thrown
#We want warnings to be thrown as warnings to be able to catch them later.
numpy.seterr(all='warn')
import warnings

#To turn input dictionary into namespace for easier access
from argparse import Namespace

#To pair-zip hyperparameter options
from itertools import product

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

#NOTE ON FITNESS WEIGHTS:
#   Weights will be used when finding the maximum fitness within the Deap library,
#   but you will see the fitness value that is return from evaluation function
#   IE: During fitness max function -> fitness * weights
#       When calling "individual.fitness.values" -> fitness / weights


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

#toolbox = base.Toolbox()

def get_params_gs():
  """Get hyperparameter pairs to run through grid search"""
  mu = [1.0, 0.5, 0.0, -0.5, -1.0]
  sigma = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005]
  alpha = [0.1, 0.3, 0.5, 0.7, 0.9]
  indpb = [0.1, 0.3, 0.5, 0.7, 0.9]
  tournsize = [2, 3, 4, 5]
  cxpb = [0.1, 0.3, 0.5, 0.7, 0.9]
  mutpb = [0.1, 0.3, 0.5, 0.7, 0.9]
  options = product(mu, sigma, alpha, indpb, tournsize, cxpb, mutpb)
  return options

def to_string():
  """Get the name of the algorithm to be put on output strings"""
  return "ga"

class Algorithm:
  def __init__(self, pool=None, problem=None, **args):
    
    args = Namespace(**args)
    
    self.toolbox = base.Toolbox()
    
    self.stats = tools.Statistics(key=self._stat_func)
    self.stats.register("avg", numpy.mean)
    self.stats.register("std", numpy.std)
    self.stats.register("min", numpy.min)
    self.stats.register("max", numpy.max)
    
    #if not pool:
    #  self.map_func = lambda func, pop, **args: [func(x, **args) for x in pop]
    #else:
    #  self.map_func = lambda func, pop, **args: pool.starmap(func, [(x, *args) for x in pop])
    
    if not pool:
      self.map_func = map
    else:
      self.map_func = pool.map
    
    if not hasattr(args, 'size'):
      self.size = 10
    else:
      self.size = args.size
      
    if not hasattr(args, 'mu'):
      args.mu = -1.0
      
    if not hasattr(args, 'sigma'):
      args.sigma = 0.001
      
    if not hasattr(args, 'alpha'):
      args.alpha = 0.9
      
    if not hasattr(args, 'indpb'):
      args.indpb = 0.9
      
    if not hasattr(args, 'tournsize'):
      args.tournsize = 4
      
    if not hasattr(args, 'debug'):
      self.debug = 0
    else:
      self.debug = args.debug
      
    if not hasattr(args, 'pop_size'):
      self.pop_size = 300
    else:
      self.pop_size = args.pop_size
      
    if not hasattr(args, 'number_generations'):
      self.num_gen = 1000
    else:
      self.num_gen = args.number_generations
      
    if not hasattr(args, 'cxpb'):
      self.cxpb = 0.5
    else:
      self.cxpb = args.cxpb
      
    if not hasattr(args, 'mutpb'):
      self.mutpb = 0.3
    else:
      self.mutpb = args.mutpb
      
    if problem == 1:
      if not hasattr(args, 'k'):
        raise ValueError("k must be given when problem 1 is being used")
      self._init_quad_opt(args.k)
    elif problem == 2:
      raise NotImplementedError("Problem 2 is not implemented yet")
    else:
      raise ValueError('parameter "problem" not provided')
      
    #Set up ways to define individuals in the population
    self.toolbox.register("attr_x", np.random.normal, 0, 1)
    self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_x, args.size)
    self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
    
    #Set up ways to change population
    self.toolbox.register("mate", tools.cxBlend, alpha=args.alpha)
    self.toolbox.register("mutate", tools.mutGaussian, mu=args.mu, sigma=args.sigma, indpb=args.indpb)
    self.toolbox.register("select", tools.selTournament, tournsize=args.tournsize)
    
  def _stat_func(self, ind):
    return ind.fitness.values[0]
    
  #Fitness evaluation methods (must return iterable)
  #Remember, we want to minimize these functions, so to hurt them we need to return
  #large positive numbers.
  #=====================================================================================
  def _evaluatate_quad_opt(self, individual):
    x = np.array([individual]).T  #x as Column vector
    
    #Have a very large fitness be returned if any value in x is nan (what we do not want)
    for i in individual:
      if np.isnan(i) or np.isinf(i):
        return (1000000000,)
    
    #There may be times where the numpy doing the math operations will have an
    #overflow warning, or another warning. In these cases, we want to return
    #the largest possible fitness since this is something unwanted.
    with warnings.catch_warnings():
      
      #We only want to filter warnings for these few lines of code, not the 
      #entire file. Having these lines within the "with" block ensures this
      #filtering only occurs here.
      warnings.filterwarnings('error')
      
      try:
        #value =  0.5 * np.matmul(np.matmul(x.T, A), x) - np.matmul(b.T, x)
        value = np.linalg.norm(np.matmul(self.A, x) - self.b, 2)
        
        #The problem will try to minimize too much (go beyond 0 for error) and become negative.
        #We do not want this, so we want a function to max out at 10 and grow smaller once you
        #go away from 0. -X^2 will give us that. We want a shallow curve though to have more 
        #differentiation around 0.000. This will make sure the algorithm does not fall short
        #once you start getting too close to 0.00. To do this, but still keep the shape, we
        #use 1.2 for the power. (the fitness value is inverted, so -1 becomes +1 or invert 
        #the graph of X^1.2, and that will be maximized)
        value = pow(value, 8)
      except Warning:
        value = 1000000000
      
    return (value,)
    
  #=====================================================================================
  
  
  #Initialization methods
  #=====================================================================================
  def _init_quad_opt(self, k):
    A, b = cqo_gen_input(k, self.size, self.debug)
    if self.size < 20 and self.debug >= 0:
      print("Exact minimizer for problem is: %r" % (np.asarray(np.matmul(np.linalg.inv(A), b))))
      
    self.A = A
    self.b = b
    self.evaluate_fitness = self._evaluatate_quad_opt
  #=====================================================================================
  
  def run(self):
    """
    Run a genetic algorithm with the given evaluation function and input parameters.
    Main portion of code for this method found from Deap example at URL:
    https://deap.readthedocs.io/en/master/overview.html
  
    Parameters
    ----------
    None
  
    Returns
    -------
    best_individual: List
      The best individual found out of all iterations
    fitness: Float
      The best_individual's fitness value
    logbook : Dictionary
      A dictionary of arrays for iterations, min, max, average, and std. dev. for each iteration.
  
    """
    pop = self.toolbox.population(n=self.pop_size)
    hof = tools.HallOfFame(25)
    logbook = tools.Logbook()
    
    # Evaluate the entire population
    fitnesses = list(self.map_func(self.evaluate_fitness, pop))
    for ind, fit in zip(pop, fitnesses):
      ind.fitness.values = fit
      ind.generation = 0
      
    record = self.stats.compile(pop) if self.stats else {}
    logbook.record(gen=0, **record)
    
  
    for g in range(self.num_gen):
      # Select the next generation individuals (with replacement)
      offspring = self.toolbox.select(pop, len(pop))
      # Clone the selected individuals (since selection only took references rather than values)
      offspring = list(map(self.toolbox.clone, offspring))
      
      # Apply crossover and mutation on the offspring
      for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < self.cxpb:
          self.toolbox.mate(child1, child2)
          del child1.fitness.values
          del child2.fitness.values
  
      for mutant in offspring:
        if random.random() < self.mutpb:
          self.toolbox.mutate(mutant)
          del mutant.fitness.values
  
      # Evaluate the individuals with an invalid fitness
      invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
      fitnesses = self.map_func(self.evaluate_fitness, invalid_ind)
      
      if self.debug >= 2:
        print("Generation %i has (min, max) fitness values: (%.3f, %.3f)" % (g, max(fitnesses)[0], min(fitnesses)[0]))
      elif self.debug == 1 and g % 10 == 0:
        print("Generation %i has (min, max) fitness values: (%.3f, %.3f)" % (g, max(fitnesses)[0], min(fitnesses)[0]))
      
      for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
        ind.generation = g + 1
      
      # The population is entirely replaced by the offspring
      pop[:] = offspring
      hof.update(pop)
      record = self.stats.compile(pop) if self.stats else {}
      logbook.record(gen = g + 1, **record)
      
    if self.debug >= 0:
      print("Convex quadratic optimization problem results:")
      if len(hof[0]) < 20:
        print("\tBest individual seen in all generations:\t%r" % (hof[0]))
        
      print("\tBest individual seen fitness value:\t\t%.3f" % (hof[0].fitness.values[0]))
      print("\tBest individual seen generation appeared in:\t%i" % (hof[0].generation))
      
    gen, min_results, max_results, avg, std = logbook.select("gen", "min", "max", "avg", "std")
    return hof[0], hof[0].fitness.values[0], {"iterations": gen, "min": min_results, "max": max_results, "avg": avg, "std": std}
  
  def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['map_func']
        return self_dict

  def __setstate__(self, state):
      self.__dict__.update(state)