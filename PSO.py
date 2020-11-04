# Adapted from https://nathanrooy.github.io/posts/2016-08-17/simple-particle-swarm-optimization-with-python/

import random
from convex_quadratic_opt import generate_input as gi
import numpy as np
import statistics 

def to_string():
    return "PSO"

class Particle:
    def __init__(self,x0):
        self.position = []        # particle position
        self.velocity_i = []          # particle velocity
        self.pos_best_i = []          # best position individual
        self.err_best_i = -1          # best error individual
        self.err_i = -1               # error individual
        self.num_dimensions = len(x0)
        self.bounds = [(-10,10) for i in range(self.num_dimensions)]
        
        for i in range(0,self.num_dimensions):
            self.velocity_i.append(random.uniform(-1,1))
            self.position.append(x0[i])
        
    # evaluate current fitness
    def evaluate(self,costFunc):
        self.err_i=costFunc(self.position)

        # check to see if the current position is an individual best
        if self.err_i < self.err_best_i or self.err_best_i==-1:
            self.pos_best_i=self.position
            self.err_best_i=self.err_i

    # update new particle velocity
    def update_velocity(self,pos_best_g):
        w=0.5       # constant inertia weight (how much to weigh the previous velocity)
        c1=1        # cognative constant
        c2=2        # social constant

        for i in range(0,self.num_dimensions):
            r1=random.random()
            r2=random.random()
            vel_cognitive=c1*r1*(self.pos_best_i[i]-self.position[i])
            #vel_cognitive=self.pos_best_i[i]-self.position[i]
            vel_social=c2*r2*(pos_best_g[i]-self.position[i])
            self.velocity_i[i]=w*self.velocity_i[i]+vel_cognitive+vel_social

    # update the particle position based off new velocity updates
    def update_position(self,bounds):
        for i in range(0,self.num_dimensions):
            self.position[i]=self.position[i]+self.velocity_i[i]

            # adjust maximum position if necessary
            if self.position[i]>self.bounds[i][1]:
                self.position[i]=self.bounds[i][1]

            # adjust minimum position if neseccary
            if self.position[i] < self.bounds[i][0]:
                self.position[i]=self.bounds[i][0]
                
class Algorithm():
    def __init__(self, **args):
        # ,costFunc,x0,bounds,num_particles,maxiter
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
        
        self.bounds = [(-10,10) for i in range(self.dimension)] 
        if args["problems"] == 1:
          self.costFunc = self.evalutate_quad_opt
        else:
          self.costFunc = self.evalutate_quad_opt
          
        # establish the swarm
        self.swarm=[]
        for i in range(0,self.num_particles):
            self.swarm.append(Particle([
                random.uniform(0, 100) for i in range(self.dimension) 
                ]))

    def run(self):
        # begin optimization loop
        for i in range(self.maxiter):
            # cycle through particles in swarm and evaluate fitness
            for j in range(0,self.num_particles):
                self.swarm[j].evaluate(self.costFunc)
                print("p_{} new cost: {}".format(j, self.swarm[j].err_i))
                # determine if current particle is the best (globally)
                if self.swarm[j].err_i < self.err_best_g or self.err_best_g == -1:
                    self.pos_best_g = self.swarm[j].position
                    self.err_best_g = self.swarm[j].err_i
                    
                smallest = min([particle.err_i for particle in self.swarm ])
                self.min_results.append(smallest)
                largest = max([particle.err_i for particle in self.swarm ])
                self.max_results.append(largest)
                self.avg.append((smallest+largest)/2)
                self.std.append(statistics.stdev([particle.err_i for particle in self.swarm ]))
                
            # cycle through swarm and update velocities and position
            for j in range(0,self.num_particles):
                self.swarm[j].update_velocity(self.pos_best_g)
                self.swarm[j].update_position(self.bounds)
            

        
        print("\tBest individual seen fitness value: {:0.3f}".format(self.err_best_g))
        
        output_dictionary = {"iterations": [i for i in range(1, self.maxiter+1)], "avg": self.avg, "min": self.min_results, "max": self.max_results, "std": self.std}
        return self.pos_best_g, self.costFunc(self.pos_best_g), output_dictionary
        # # print final results
        # print 'FINAL:'
        # print pos_best_g
        # print err_best_g

    # optimization function 1
    def evalutate_quad_opt(self, individual):
        x = np.array(individual, dtype=float)
        #value = 0.5 * np.matmul(np.matmul(x.T, self.A), x) - np.matmul(self.b.T, x)
        value = np.linalg.norm(np.matmul(self.A, x) - self.b, 2)
        return value

