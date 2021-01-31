#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 14:58:39 2020

@author: jamilgafur
"""
from convex_quadratic_opt import generate_input as gi
from convex_quadratic_opt import nonconvex_generate_input as gnci
import numpy as np
import statistics
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.tri as tri
from itertools import product


def get_params_gs():
    """Get hyperparameter pairs to run through grid search"""
    gc = [1.0, 0.5, 0.0, -0.5, -1.0]
    gd = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005]
    options = product(gc, gd)
    return options


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

        for i in range(0, self.num_dimensions):
            if np.random.random() > .5:
                self.position.append(np.random.normal(0, 2))
            else:
                self.position.append(np.random.normal(0, 2) * -1)

    # evaluate current fitness
    def evaluate(self, costFunc):
        self.cost = costFunc(self.position)

    def calc_inertia(self, smallest, largest):
        self.inertia = np.subtract(self.cost, largest) / np.subtract(smallest, largest) + .0001
        return self.inertia

    def calc_mass(self, total_inertia):
        self.mass = np.divide(self.inertia, total_inertia)

    def calc_force(self, other_particle, gravity):
        numerator = np.multiply(self.mass, other_particle.mass)
        denominator = np.linalg.norm(np.subtract(self.position, other_particle.position)) + .001
        self.force += np.random.random() * gravity * (numerator / denominator) * (
            np.subtract(other_particle.position, self.position))

    def calc_acceleration(self):
        self.accel = np.divide(self.force, self.mass)

    def calc_velocity(self):
        self.velocity = np.random.uniform(0, 1) * np.add(self.velocity, self.accel)

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
                                                                                  round(self.cost, 7),
                                                                                  round(self.inertia, 7),
                                                                                  round(self.mass, 7),
                                                                                  round(sum(self.force), 7),
                                                                                  round(sum(self.accel), 7),
                                                                                  round(sum(self.velocity), 7))
            print(tableformat)
        self.position = np.add(self.position, self.velocity).tolist()


class Algorithm:
    def __init__(self, **args):
        # ========command arguments===========
        self.debug = args["debug"]
        self.dimension = args["size"]
        self.num_particles = args["num_particles"]
        self.maxiter = args["number_generations"]
        self.cost_var = args["problems"]
        self.gc = args["gc"]
        self.gd = args["gd"]
        self.k = args['k']
        # ============algorithm inputs========
        self.swarm = []
        self.best_iteration = 0
        self.best_cost = None
        self.best_cost_location = []
        self.avg = []
        self.min_results = []
        self.max_results = []
        self.std = []
        self.history_loc = []
        self.solution = []
        self.contor_lvl = args["cl"]
        # =========== search problem==========
        if self.cost_var == 0:
            self.costFunc = self.evalutate_quad_opt
            self.A, self.b = gi(args['k'], args["size"], args["debug"])
            self.solution = np.matmul(np.linalg.inv(self.A), self.b)
        else:
            self.costFunc = self.evalutate_noncon_opt
            self.solution = [0]  # temp
            self.m = args['ncm']
            self.M = args['ncM']
            self.b = args['ncb']
            self.alpha, self.beta, self.sigma = gnci(args['ncm'], args['ncM'], args['ncb'], self.dimension)

        if self.debug and self.dimension == 2:
            self.fig = plt.figure()
            self.ax = plt.axes()
            self.line, = self.ax.plot([], [], 'o', color='black')

        # =========== populate swarm===========
        for i in range(0, self.num_particles):
            self.swarm.append(Particle(self.dimension, i, self.debug))

    def init_animation(self):
        """
         initalizes the visualzation
        """
        self.line.set_data([], [])
        return self.line,

    def animate(self, time):
        """
        updates the gif for a specific timestep
        """
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
            self.gravity = self.gc * np.divide(i, self.maxiter) ** self.gd

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

            self.min_results.append(round(smallest, 3))
            self.max_results.append(round(largest, 3))
            self.avg.append(round(sum([particle.cost for particle in self.swarm]) / self.num_particles, 3))
            self.std.append(statistics.stdev([particle.cost for particle in self.swarm]))
            # =======logging========
            if self.debug > 0:
                print("\t-----")
                print("\tbest:{}".format(self.best_cost))
                print("\tsmall:{} \tlarge:{}".format(smallest, largest))
                print("\t-----")
                # =======visualization========
                if self.dimension == 2:
                    time = []
                    for particle in self.swarm:
                        time.append(particle.position)
                    self.history_loc.append(time)
        # =======generate gif===========
        if self.debug > 0 and (self.dimension == 2):
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

            ngridx = 100
            ngridy = 100
            xi = np.linspace(-2, 2, ngridx)
            yi = np.linspace(-2, 2, ngridy)

            Xi, Yi = np.meshgrid(xi, yi)
            zi = interpolator(Xi, Yi)

            plt.contourf(Xi, Yi, zi, self.contor_lvl, cmap='RdGy')
            plt.colorbar()
            if self.cost_var == 0:
                anim.save(
                    'GSA_prob_{}_pop_{}_k_{}_n_{}_gc_{}_gd_{}_iter_{}.gif'.format(self.cost_var, self.num_particles,
                                                                                  self.k, self.dimension, self.gc,
                                                                                  self.gd, self.maxiter),
                    writer='imagemagick')
            else:
                anim.save('GSA_prob_{}_pop_{}_n_{}_gc_{}_gd_{}_iter_{}_ncm_{}_ncM_{}_ncb_{}.gif'.format(self.cost_var,
                                                                                                        self.num_particles,
                                                                                                        self.dimension,
                                                                                                        self.gc,
                                                                                                        self.gd,
                                                                                                        self.maxiter,
                                                                                                        self.m, self.M,
                                                                                                        self.b, ),
                          writer='imagemagick')

            # print("\nsolution: {}\nsolution_cost:{}\n\n".format(self.best_cost_location, self.costFunc(self.best_cost_location)))
        output_dictionary = {"iterations": [i for i in range(self.maxiter)], "min": self.min_results,
                             "max": self.max_results, "avg": self.avg, "std": self.std}
        return self.best_cost_location, self.costFunc(self.best_cost_location), output_dictionary

    # optimization function 1
    def evalutate_quad_opt(self, individual):
        x = np.array(individual, dtype=float)
        value = np.linalg.norm(np.matmul(self.A, x) - self.b, 2)
        return value

    # optimization function 2
    def evalutate_noncon_opt(self, x):
        x = np.array(x, dtype=float)
        # 1/2 z^2
        front = .5 * np.multiply(x, x)
        # inner = Beta_i *z + sigma_i
        inner = np.array([np.add(np.multiply(self.beta[i], x), self.sigma[i]) for i in range(0, len(x))])
        # inner = cos(inner)
        inner = np.cos(inner)
        # inner = alpha_i * cos(inner)_i
        inner = np.multiply(self.alpha, inner)
        value = np.add(front, inner)
        return np.sum(value)
