#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 14:58:39 2020

@author: jamilgafur
"""
from convex_quadratic_opt import generate_input as gi
import numpy as np
import statistics 

def to_string():
    return "GSA"

class Particle:
    def __init__(self, dimension, name, debug):
        self.debug = debug
        self.name = name
        self.position = []
        self.num_dimensions = dimension
        self.force = [0 for i in range(self.num_dimensions)]
        self.accel = [0 for i in range(self.num_dimensions)]
        self.inertia = 0
        self.velocity = [0 for i in range(self.num_dimensions)]
        self.gravity = 0
        self.mass = 0
        self.cost = None
        
        for i in range(0,self.num_dimensions):
            self.position.append(np.random.normal(0,1))
        
    # evaluate current fitness
    def evaluate(self,costFunc):
        self.cost = costFunc(self.position)
                
    def calc_inertia(self, smallest, largest):
        self.inertia = np.subtract(self.cost, largest) / np.subtract(smallest, largest) + .0001
        return self.inertia
        
    def calc_mass(self, total_inertia):
         self.mass = np.divide(self.inertia, total_inertia) 
         
    def calc_force(self,other_particle,gravity ):
        numerator = np.multiply(self.mass, other_particle.mass)
        denominator = np.linalg.norm(np.subtract(self.position, other_particle.position )) + .001
        self.force += np.random.random() * gravity * (numerator / denominator) * (np.subtract(other_particle.position , self.position))
            
    def calc_acceleration(self):
        self.accel = np.divide(self.force, self.mass)
        
    def calc_velocity(self):
        self.velocity = np.random.uniform(0,1) * np.add(self.velocity,  self.accel)
        
    # update the particle position based off new velocity updates
    def update_position(self):
        if self.debug > 0:
            print("{:5}\t{:10}\t{:10}\t{:10}\t{:10}\t{:10}\t{:10}".format("name",
                    "cost",
                    "inertia",
                    "mass",
                    "force",
                    "accel",
                    "velocity"))
            tableformat = "{:5}\t{:10}\t{:10}\t{:10}\t{:10}\t{:10}\t{:10}".format(self.name,
                                                            round(self.cost,7),
                                                            round(self.inertia,7),
                                                            round(self.mass,7),
                                                            round(sum(self.force),7),
                                                            round(sum(self.accel),7),
                                                            round(sum(self.velocity),7))
            print(tableformat)
        self.position = np.add(self.position,self.velocity).tolist()
                
class Algorithm():
    def __init__(self, **args):
        self.debug = args["debug"]

        self.A = None
        self.b = None
        self.A, self.b = gi(args["k"], args["size"], args["debug"])
        self.dimension = args["size"]
        self.best_cost = None                   # best error for group
        self.best_cost_location = []                   # best position for group
        self.best_iteration = 0
        self.num_particles = args["pop_size"]
        self.maxiter = args["number_generations"]
        self.avg  = [] 
        self.min_results = []
        self.max_results = []
        self.std = []
        self.gravity = 0
        
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
            self.gravity = np.exp( np.divide(i, self.maxiter)**.5  )  

            # cycle through particles in swarm and evaluate fitness
            for particle in self.swarm:
                particle.evaluate(self.costFunc)

            
            smallest = min([particle.cost for particle in self.swarm])
            largest = max([particle.cost for particle in self.swarm])
            
            if self.best_cost == None or smallest < self.best_cost:
                self.best_cost = smallest
                self.best_iteration = i
                    
            total_intertia = 0
            for particle in self.swarm:
                if particle.cost == smallest:
                    self.best_cost_location = particle.position
                    
                total_intertia += particle.calc_inertia(smallest, largest)  
            
            for particle in self.swarm:
                particle.calc_mass(total_intertia)
                particle.force = [0 for i in range(self.dimension)]
        
            for particle in self.swarm:
                for other_particle in self.swarm:
                    if particle.position != other_particle.position:
                        particle.calc_force(other_particle, self.gravity)
                
            for particle in self.swarm:
                particle.calc_acceleration()
                particle.calc_velocity()
                particle.update_position()
                    

            self.min_results.append(round(smallest,3))
            self.max_results.append(round(largest,3))
            self.avg.append(round(sum([particle.cost for particle in self.swarm])/self.num_particles, 3))
            self.std.append(statistics.stdev([particle.cost for particle in self.swarm ]))
            if self.debug > 0:
                print("\t-----")
                print("\tbest:{}".format(self.best_cost))
                print("\tsmall:{} \tlarge:{}".format(smallest,largest))
                print("\t-----")

        print(self.min_results)
        print("\tBest individual seen fitness value: {:0.3f}".format(self.best_cost))
        output_dictionary = {"iterations": [i for i in range(self.maxiter)], "min": self.min_results, "max": self.max_results, "avg": self.avg, "std": self.std}
        return self.best_cost_location, self.costFunc(self.best_cost_location), output_dictionary


    # optimization function 1
    def evalutate_quad_opt(self, individual):
        x = np.array(individual, dtype=float)
        #value = 0.5 * np.matmul(np.matmul(x.T, self.A), x) - np.matmul(self.b.T, x)
        value = np.linalg.norm(np.matmul(self.A, x) - self.b, 2)
        return value


