# Adapted from https://nathanrooy.github.io/posts/2016-08-17/simple-particle-swarm-optimization-with-python/

import random
from convex_quadratic_opt import generate_input as gi
import numpy as np
import statistics 

def to_string():
    return "PSO"

class Particle:
    def __init__(self,dimension, name, debug):
        self.debug = debug
        self.name = name
        self.position = []        # particle position
        self.velocity_i = []          # particle velocity
        self.pos_best_i = []          # best position individual
        self.err_best_i = -1          # best error individual
        self.cost_i = -1               # error individual
        self.num_dimensions =         self.debug = debug

        self.velocity_i = [0 for i in range(self.num_dimensions)]
        self.position = []
        for i in range(0,self.num_dimensions):
            self.position.append(np.random.normal(0,1))
        
    # evaluate current fitness
    def evaluate(self,costFunc):
        self.cost_i = costFunc(self.position)

        # check to see if the current position is an individual best
        if self.err_best_i== -1 or self.cost_i < self.err_best_i:
            self.pos_best_i=self.position
            self.err_best_i=self.cost_i

    # update new particle velocity
    def update_velocity(self,pos_best_g):
        w=.05 # constant inertia weight (how much to weigh the previous velocity)
        c1=1 * random.random()        # cognative constant
        c2=2 * random.random()        # social constant
        
        vel_cognitive=np.multiply(c1,np.subtract(self.pos_best_i, self.position))
        vel_social   =np.multiply(c2,np.subtract(pos_best_g , self.position))
        self.velocity_i =np.multiply( w,self.velocity_i) + vel_cognitive+vel_social

    # update the particle position based off new velocity updates
    def update_position(self):
       if self.debug > 0:
          print("{:5}\t{:10}".format("name",
                    "cost"))
          tableformat = "{:5}\t{:10}".format(self.name, round(self.cost_i,7))
          print(tableformat)
          
       self.position= np.add(self.position,self.velocity_i)
                
class Algorithm():
    def __init__(self, **args):
        self.debug = args["debug"]
        self.A = None
        self.b = None
        self.A, self.b = gi(args["k"], args["size"], args["debug"])

        self.dimension = args["size"]
        self.err_best_g = -1                   # best error for group
        self.pos_best_g = []                   # best position for group
        self.num_particles = args["pop_size"]
        self.maxiter = args["number_generations"]
        self.avg  = [] 
        self.min_results = []
        self.max_results = []
        self.std = []
        
        if args["problems"] == 1:
          self.costFunc = self.evalutate_quad_opt
        else:
          self.costFunc = self.evalutate_quad_opt
          
        # establish the swarm
        self.swarm=[]
        for i in range(0,self.num_particles):
            self.swarm.append(Particle(self.dimension, i, self.debug))

    def run(self):
        # begin optimization loop
        for i in range(self.maxiter):
            # cycle through particles in swarm and evaluate fitness
            for particle in self.swarm:
                particle.evaluate(self.costFunc)
                # determine if current particle is the best (globally)
                if self.err_best_g == -1 or particle.cost_i < self.err_best_g :
                    self.pos_best_g = particle.position
                    self.err_best_g = particle.cost_i
                    

            for particle in self.swarm:
                particle.update_velocity(self.pos_best_g)
                particle.update_position()
            
            smallest = min([particle.cost_i for particle in self.swarm ])
            self.min_results.append(smallest)
            largest = max([particle.cost_i for particle in self.swarm ])
            self.max_results.append(largest)
            self.avg.append(sum([particle.cost_i for particle in self.swarm])/self.num_particles)
            self.std.append(statistics.stdev([particle.cost_i for particle in self.swarm ]))



        
        print("\tBest individual seen fitness value: {:0.3f}".format(self.err_best_g))
        
        output_dictionary = {"iterations": [i for i in range(self.maxiter)], "min": self.min_results, "max": self.max_results, "avg": self.avg, "std": self.std}
        return self.pos_best_g, self.costFunc(self.pos_best_g), output_dictionary


    # optimization function 1
    def evalutate_quad_opt(self, individual):
        x = np.array(individual, dtype=float)
        #value = 0.5 * np.matmul(np.matmul(x.T, self.A), x) - np.matmul(self.b.T, x)
        value = np.linalg.norm(np.matmul(self.A, x) - self.b, 2)
        return value


