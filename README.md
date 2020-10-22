# Setup for algorithms
Each algorithm needs to have the following setup to work with the main file.
1. The class inside the algorithm file needs to be named ```Algorithm```
2. The method used to start the algorithm needs to be called ```run``` with no input parameters (besides self)
3. There should be a method outside the Algorithm class that is named ```get_params_gs```
    * This method will return a product of all possible hyperparameter values used for Grid Search
    * It should not take any input
    * Needs to return an object of type itertools.product
4. The ```Algorithm``` class needs to be pickleable
    * NOTE: You may not have to do anything to make the class pickleable. All that is required is that no unpickleable objects or modules are used or imported within the class
    * IF an unpickleable object or module is required, then look at genetic algorithm file to see how to make the class pickleable. It requires the methods ```__getstate__``` and ```__setstate__```
5. Have a method outside the ```Algorithm``` class called ```to_string``` that takes in no parameters. It will return the string that is the name of the algorithm to put on output files.

# Module interface definition
Below is the actual module interface definition. Any algorithm should start with this as the baseline and build up from it.
```python
import numpy as np 

#Imports for problems
from convex_quadratic_opt import generate_input as cqo_gen_input

#To turn input dictionary into namespace for easier access
from argparse import Namespace

#To pair-zip hyperparameter options
from itertools import product

def get_params_gs():
  """Get hyperparameter pairs to run through grid search"""
  parameter_1 = [...]
  parameter_2 = [...]
  #...
  options = product(parameter_1, parameter_2, ...)
  return options

def to_string():
  """Get the name of the algorithm to be put on output strings"""
  return "alg_name"

class Algorithm:
  def __init__(self, problem=None, **args):
    
    args = Namespace(**args)
    
    if not hasattr(args, 'size'):
      self.size = 10
    else:
      self.size = args.size
      
    #Set algorithm parameters to self object here...
    #Here is an example
    if not hasattr(args, 'parameter_1'):
      self.parameter_1 = 10
    else:
      self.parameter_1 = args.parameter_1
      
    if not hasattr(args, 'debug'):
      self.debug = 0
    else:
      self.debug = args.debug
      
    if problem == 1:
      if not hasattr(args, 'k'):
        raise ValueError("k must be given when problem 1 is being used")
      self._init_quad_opt(args.k)
    elif problem == 2:
      raise NotImplementedError("Problem 2 is not implemented yet")
    else:
      raise ValueError('parameter "problem" not provided')
      
  
  #Initialization methods
  #=====================================================================================
  def _init_quad_opt(self, k):
    A, b = cqo_gen_input(k, self.size, self.debug)
    if self.size < 20 and self.debug >= 0:
      print("Exact minimizer for problem is: %r" % (np.asarray(np.matmul(np.linalg.inv(A), b))))
      
    self.A = A
    self.b = b
    #Initialize cost function for problem 1 here...
  #=====================================================================================
  
  def run(self):
    """
  
    Parameters
    ----------
    None
  
    Returns
    -------
    best_individual: List
      The best individual found out of all iterations
     cost: Float
      The best_individual's cost value
    logbook : Dictionary
      A dictionary of arrays for iterations, min, max, average, and std. dev. for each iteration.
  
    """
    best_individual = #...
    best_individual_cost = #...
    iterations = [0, 1, 2, 3, 4, ...]
    min_results = #[...]
    max_results = #[...]
    avg = #[...]
    std = #[...]
    output_dictionary = {"iterations": iterations, "min": min_results, "max": max_results, "avg": avg, "std": std}
    return best_individual, best_individual_cost, output_dictionary
```

Now, to import new algorithms into ```main.py```, use the following snippet as a guide:
```python
from argparse import ArgumentParser

import random
import numpy.matlib 
import numpy as np 

#For parallel computing
import multiprocessing
import signal

#Imports for the algorithms
import genetic_algorithm_v2 as ga
import algorithm as agl_name

#For graphing
import matplotlib.pyplot as plt

#For outputing csv
import csv

...

def main():
  parser = build_parser()
  options = parser.parse_args()
  random.seed(options.seed)
  np.random.seed(options.seed)
  
  #Add imported algorithm modules to this list to have them be used.
  for alg in [ga, alg_name]:
    setup_alg(options, alg)
```

The biggest things to note are the importing of the algorithm module and adding the module to the algorithms array in the main method. Say we named the algorithm file ```algorithm.py```, and so that is what we want to import.
