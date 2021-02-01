#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 14:31:37 2020

@author: jamilg
"""

# Adapted from https://nathanrooy.github.io/posts/2016-08-17/simple-particle-swarm-optimization-with-python/

from convex_quadratic_opt import generate_input as gi
import numpy as np
import statistics
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.tri as tri


def to_string():
    return "RAN"


class Particle:
    def __init__(self, dimension, name, debug, cost_func):
        self.debug = debug
        self.name = name
        self.num_dimensions = dimension
        self.position = []
        for i in range(0, self.num_dimensions):
            if np.random.random() > .5:
                self.position.append(np.random.normal(0, 1))
            else:
                self.position.append(np.random.normal(0, 1) * -1)

        self.cost_func = cost_func
        self.cost = cost_func(self.position)


class Algorithm:
    def __init__(self, **args):
        self.debug = args["debug"]
        self.A = None
        self.b = None
        self.k = args["k"]
        self.A, self.b = gi(self.k, args["size"], args["debug"])
        self.cost_var = args["problems"]

        self.vel_weight = args["vw"]
        self.social_weight = args["cw"]
        self.cognitive_weight = args["sw"]
        self.dimension = args["size"]
        self.num_particles = args["num_particles"]
        self.maxiter = args["number_generations"]
        self.avg = []
        self.min_results = []
        self.max_results = []
        self.std = []
        self.solution_position = None
        self.solution_cost = 1000
        self.contor_lvl = args['cl']
        self.sample_points = args['rsn']
        if args["problems"] == 0:
            self.costFunc = self.evalutate_quad_opt
            self.solution = np.asarray(np.matmul(np.linalg.inv(self.A), self.b))
        else:
            self.costFunc = self.evalutate_quad_opt
            self.solution = np.asarray(np.matmul(np.linalg.inv(self.A), self.b))

        if self.debug and len(self.A) == 2:
            self.fig = plt.figure()
            self.ax = plt.axes()
            self.line, = self.ax.plot([], [], 'o', color='black')

        # establish the swarm
        self.history_loc = []
        self.swarm = []
        time = []
        for i in range(0, self.num_particles):
            self.swarm.append(Particle(self.dimension, i, self.debug, self.costFunc))
            time.append(self.swarm[i].position)
        self.history_loc.append(time)

    def init_animation(self):
        self.line.set_data([], [])
        return self.line,

    def animate(self, time):
        x = []
        y = []
        for i in self.history_loc[time]:
            x.append(i[0])
            y.append(i[1])

        self.line.set_data(x, y)

        return self.line,

    def run(self):
        # begin optimization loop
        for i in range(self.maxiter):
            smallest = min([particle.cost for particle in self.swarm])
            self.min_results.append(smallest)
            largest = max([particle.cost for particle in self.swarm])
            self.max_results.append(largest)
            self.avg.append(sum([particle.cost for particle in self.swarm]) / self.num_particles)
            self.std.append(statistics.stdev([particle.cost for particle in self.swarm]))
            if self.debug > 0:
                print("\t-----")
                # err_best_g is never defined
                # print("\tbest:{}".format(self.err_best_g))
                print("\tsmall:{} \tlarge:{}".format(smallest, largest))
                print("\t-----")
                if len(self.A) == 2:
                    time = []
                    for particle in self.swarm:
                        time.append(particle.position)
                    self.history_loc.append(time)
            # update position
            for particle in self.swarm:
                # try 100 different points:
                for i in range(self.sample_points):
                    rv = np.random.rand(self.dimension)
                    for i in range(len(rv)):
                        if np.random.random() > .5:
                            rv[i] = -1 * rv[i]

                    temp_location = np.add(particle.position, rv)
                    temp_cost = self.costFunc(temp_location)
                    if temp_cost <= particle.cost:
                        particle.position = temp_location
                        particle.cost = temp_cost
                    if particle.cost < self.solution_cost:
                        self.solution_position = particle.position
                        self.solution_cost = particle.cost

        if self.debug > 0 and (len(self.A) == 2):
            self.ax.axis([-2, 2, -2, 2])
            anim = FuncAnimation(self.fig, self.animate, init_func=self.init_animation, frames=len(self.history_loc),
                                 interval=500, blit=True)

            npts = 10000
            x = np.random.uniform(-2, 2, npts)
            y = np.random.uniform(-2, 2, npts)
            z = []
            for r, c in zip(x, y):
                z.append(self.costFunc([r, c]).item(0))

            triang = tri.Triangulation(x, y)
            interpolator = tri.LinearTriInterpolator(triang, z)

            ngridx = 10000
            ngridy = 10000
            xi = np.linspace(-2, 2, ngridx)
            yi = np.linspace(-2, 2, ngridy)

            Xi, Yi = np.meshgrid(xi, yi)
            zi = interpolator(Xi, Yi)

            plt.contourf(Xi, Yi, zi, self.contor_lvl, cmap='RdGy')
            plt.colorbar()
            anim.save('PSO_k_{}_prob_{}_pop_{}.gif'.format(self.k, self.cost_var, self.num_particles),
                      writer='imagemagick')

            print("\nsolution: {}\nsolution_cost:{}".format(self.solution_position, self.costFunc(self.solution_position)))
        output_dictionary = {"iterations": [i for i in range(self.maxiter)], "min": self.min_results,
                             "max": self.max_results, "avg": self.avg, "std": self.std}
        return self.solution_position, self.costFunc(self.solution_position), output_dictionary, \
               [particle.cost for particle in self.swarm]

    # optimization function 1
    def evalutate_quad_opt(self, individual):
        x = np.array(individual, dtype=float)
        value = np.linalg.norm(np.matmul(self.A, x) - self.b, 2)
        return value
