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
        self.position = []        # particle position
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
        
        
    def calc_inertia(self, best, worst):
        self.inertia = np.subtract(self.cost+ np.finfo('float32').eps, worst) / np.subtract(best, worst)
        return self.inertia
        
    def calc_mass(self, total_inertia):
        self.mass = np.divide(self.inertia, total_inertia)
        
    def update_gravity(self, current_iteration, total_iteration):
        self.gravity = np.exp(-1 * (np.divide(current_iteration, total_iteration) ))

    def calc_force(self,other_particle):
        if not self.position == other_particle.position:
            numerator = np.multiply(other_particle.mass, self.mass)
            denominator = np.sqrt(np.subtract(self.position , other_particle.position) ** 2) + np.finfo('float32').eps
            self.force += self.gravity * (numerator / denominator) * np.subtract(other_particle.position , self.position)
            self.force *=  np.random.uniform(0,1)
        
        
    def calc_acceleration(self):
        self.accel = np.divide(self.force, self.mass)
        
    def calc_velocity(self):
        self.velocity = np.random.uniform(0, 1) * np.add(self.velocity ,self.accel)
        
    # update the particle position based off new velocity updates
    def update_position(self):
        if self.debug > 0:
            tableformat = "{:5}\t{:10}\t{:10}\t{:10}\t{:10}\t{:10}\t{:10}\t{:10}".format(self.name,
                                                            round(self.cost,7),
                                                            round(self.inertia,7),
                                                            round(self.mass,7),
                                                            round(self.gravity,7),
                                                            round(self.force[0],7),
                                                            round(self.accel[0],7),
                                                            round(self.velocity[0],7))
            print(tableformat)
        self.position =  np.add(self.position,self.velocity).tolist()
                
class Algorithm():
    def __init__(self, **args):
        self.debug = args["debug"]
        if self.debug > 0:
            print("{:2}{:10}{:10}{:10}{:10}{:10}{:10}{:10}\n".format("name",
                                "cost",
                                "inertia",
                                "mass",
                                "gravity",
                                "force",
                                "accel",
                                "velocity"))
        self.A = None
        self.b = None
        self.A, self.b = gi(args["k"], args["size"], args["debug"])
        self.dimension = args["size"]
        self.best_cost = None                   # best error for group
        self.best_cost_location = []                   # best position for group
        self.best_iteration = 0
        self.worse_cost = None                   # best error for group
        self.worse_cost_location = []                   # best position for group
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
                if self.best_cost == None or particle.cost < self.best_cost:
                    self.best_cost = particle.cost
                    self.best_cost_location = particle.position.copy()
                    self.best_iteration = i
                    
                if self.worse_cost == None or particle.cost > self.worse_cost:
                    self.worse_cost_location = particle.position.copy()
                    self.worse_cost = particle.cost
                    
            total_intertia = 0
            for particle in self.swarm:
                total_intertia += particle.calc_inertia(self.best_cost, self.worse_cost)  
                
            for particle in self.swarm:
                particle.calc_mass(total_intertia)
                particle.update_gravity(i, self.maxiter)

            for particle in self.swarm:
                for other_particle in self.swarm:
                    if particle.position != other_particle.position:
                        particle.calc_force(other_particle)
                particle.calc_acceleration()
                particle.calc_velocity()
                particle.update_position()
                    

            smallest = max([particle.cost for particle in self.swarm])
            self.min_results.append(smallest)
            largest = min([particle.cost for particle in self.swarm])
            self.max_results.append(largest)
            self.avg.append((smallest+largest)/self.num_particles)
            self.std.append(statistics.stdev([particle.cost for particle in self.swarm ]))
            
            if self.debug > 0:
                print("\t-----")
                print("\tbest:{} \tworse:{}".format(self.best_cost,self.worse_cost))
                print("\t-----")

        
        print("\tBest individual seen fitness value: {:0.3f}".format(self.best_cost))
        output_dictionary = {"iterations": [i for i in range(1, self.maxiter+1)], "max": self.max_results, "avg": self.avg, "min": self.min_results, "std": self.std}
        return self.best_cost_location, self.costFunc(self.best_cost_location), output_dictionary


    # optimization function 1
    def evalutate_quad_opt(self, individual):
        x = np.array(individual, dtype=float)
        #value = 0.5 * np.matmul(np.matmul(x.T, self.A), x) - np.matmul(self.b.T, x)
        value = np.linalg.norm(np.matmul(self.A, x) - self.b, 2)
        return value


