#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 14:31:37 2020

@author: jamilg
"""

# Adapted from https://nathanrooy.github.io/posts/2016-08-17/simple-particle-swarm-optimization-with-python/

from convex_quadratic_opt import generate_input as gi, generate_solution_nonconvex
from convex_quadratic_opt import nonconvex_generate_input as gnci
from convex_quadratic_opt import f_vect
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
    def __init__(self, problem, **args):
        self.debug = args["debug"]
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

        # ========problem input=======
        if problem == 0:
            key_problem1 = "0_k{}_n{}_b{}_m{}_M{}".format(args['k'], args['size'], args['ncb'], args['ncm'],
                                                          args['ncM'])
            self.costFunc = self.evaluate_quad_opt
            self.A = args['dic'][key_problem1][0]
            self.b = args['dic'][key_problem1][1]
            self.solution = args['dic'][key_problem1][2]
        elif problem == 1:
            key_problem2 = "1_k{}_n{}_b{}_m{}_M{}".format(args['k'], args['size'], args['ncb'], args['ncm'],
                                                          args['ncM'])
            self.costFunc = self.evaluate_nonconvex_optimizer
            self.Q = args['dic'][key_problem2][0]
            self.alpha = args['dic'][key_problem2][1]
            self.beta = args['dic'][key_problem2][2]
            self.gamma = args['dic'][key_problem2][3]
            self.solution = args['dic'][key_problem2][4]
        else:
            raise ValueError('parameter "problem" not provided')

        if self.debug and self.dimension == 2:
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
            # update position
            for particle in self.swarm:
                # try different points:
                for _ in range(self.sample_points):
                    ran_vector = np.random.rand(self.dimension)
                    for ran_value in range(len(ran_vector)):
                        if np.random.random() > .5:
                            ran_vector[ran_value] = -1 * ran_vector[ran_value]

                    temp_location = np.add(particle.position, ran_vector)
                    temp_cost = self.costFunc(temp_location)
                    if temp_cost <= particle.cost:
                        particle.position = temp_location
                        particle.cost = temp_cost
                    if particle.cost < self.solution_cost:
                        self.solution_position = particle.position
                        self.solution_cost = particle.cost

            loss_values = [particle.cost for particle in self.swarm]
            smallest = min(loss_values)
            self.min_results.append(smallest)
            largest = max(loss_values)
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

            print("\nsolution: {}\nsolution_cost:{}".format(self.solution_position,
                                                            self.costFunc(self.solution_position)))
        output_dictionary = {"iterations": [i for i in range(self.maxiter)], "min": self.min_results,
                             "max": self.max_results, "avg": self.avg, "std": self.std}

        diffs = []
        for particle in self.swarm:
            diffs.append(np.sqrt(np.square(np.subtract(self.solution, particle.position))).mean())

        # print("got: {}\tcost:{}".format(self.solution_position, self.costFunc(self.solution_position)))
        # print("sol: {}\tcost:{}".format(self.solution, self.costFunc(self.solution)))

        return self.solution_position, self.solution_cost, output_dictionary, loss_values, diffs

    # optimization function 1
    def evaluate_quad_opt(self, individual):
        x = np.array(individual, dtype=float)
        value = np.linalg.norm(np.matmul(self.A, x) - self.b, 2)
        return value

    # optimization function 2
    def evaluate_nonconvex_optimizer(self, x):
        return f_vect(x, self.Q, self.alpha, self.beta, self.gamma)
