# Adapted from https://nathanrooy.github.io/posts/2016-08-17/simple-particle-swarm-optimization-with-python/

from convex_quadratic_opt import generate_input as gi
import numpy as np
import statistics 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.tri as tri

def to_string():
    return "PSO"

class Particle:
    def __init__(self,dimension, name, debug, social, cognitive, vel):
        self.debug = debug
        self.name = name
        self.velocity_i = []          # particle velocity
        self.pos_best_i = []          # best position individual
        self.err_best_i = -1          # best error individual
        self.cost_i = -1               # error individual
        self.num_dimensions = dimension
        self.debug = debug
        self.s_weight = social
        self.c_weight = cognitive
        self.v_weight = vel
        self.velocity_i = [0 for i in range(self.num_dimensions)]
        self.position = []
        for i in range(0,self.num_dimensions):
            if np.random.random() > .5:
                self.position.append(np.random.normal(0,1))
            else:
                self.position.append(np.random.normal(0,1)*-1)
        
    # evaluate current fitness
    def evaluate(self,costFunc):
        self.cost_i = costFunc(self.position)

        # check to see if the current position is an individual best
        if self.err_best_i== -1 or self.cost_i < self.err_best_i:
            self.pos_best_i=self.position
            self.err_best_i=self.cost_i

    # update new particle velocity
    def update_velocity(self,pos_best_g):
        for i in range(self.num_dimensions):
            r1=np.random.random()
            r2=np.random.random()

            vel_cognitive=self.c_weight*r1*(self.pos_best_i[i]-self.position[i])
            vel_social=self.s_weight*r2*(pos_best_g[i]-self.position[i])
            self.velocity_i[i]=self.v_weight*self.velocity_i[i]+vel_cognitive+vel_social
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
        self.k = args["k"]
        self.A, self.b = gi(self.k , args["size"], args["debug"])
        self.cost_var = args["problems"]

        self.vel_weight =  args["vw"]
        self.social_weight =  args["cw"]
        self.cognitive_weight =  args["sw"]
        self.dimension = args["size"]
        self.err_best_g = None          # best error for group
        self.pos_best_g = []            # best position for group
        self.num_particles = args["pop_size"]
        self.maxiter = args["number_generations"]
        self.avg  = [] 
        self.min_results = []
        self.max_results = []
        self.std = []
        self.solution = []
        if args["problems"] == 1:
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
        self.swarm=[]
        time = []
        for i in range(0,self.num_particles):
            self.swarm.append(Particle(self.dimension, i, self.debug, self.social_weight, self.cognitive_weight, self.vel_weight))
            time.append(self.swarm[i].position)
        self.history_loc.append(time)
        
    def init_animation(self):
        self.line.set_data([],[])
        return self.line, 

    def animate(self, time):
        x = []
        y =[]
        for i in self.history_loc[time]:
            x.append(i[0])
            y.append(i[1])
        
        self.line.set_data(x,y)
        
        return self.line, 
    def run(self):
        # begin optimization loop
        for i in range(self.maxiter):
            # cycle through particles in swarm and evaluate fitness
            for particle in self.swarm:
                particle.evaluate(self.costFunc)
                # determine if current particle is the best (globally)
                if self.err_best_g == None or particle.cost_i < self.err_best_g :
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
            if self.debug > 0:
                print("\t-----")
                print("\tbest:{}".format(self.err_best_g))
                print("\tsmall:{} \tlarge:{}".format(smallest,largest))
                print("\t-----")
                if len(self.A) == 2:
                    time = []
                    for particle in self.swarm:
                        time.append(particle.position)
                    self.history_loc.append(time)
                    
        if self.debug > 0 and (len(self.A) == 2):
            self.ax.axis([-2,2,-2,2])
            anim = FuncAnimation(self.fig, self.animate, init_func=self.init_animation, frames=len(self.history_loc), interval=500, blit=True)
            
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

            plt.contourf(Xi, Yi, zi, cmap='RdGy')
            plt.colorbar()
            anim.save('PSO_k_{}_prob_{}_pop_{}.gif'.format(self.k, self.cost_var, self.num_particles), writer='imagemagick')
        
        print("\nsolution: {}\nsolution_cost:{}".format(self.pos_best_g, self.costFunc(self.pos_best_g)))
        output_dictionary = {"iterations": [i for i in range(self.maxiter)], "min": self.min_results, "max": self.max_results, "avg": self.avg, "std": self.std}
        return self.pos_best_g, self.costFunc(self.pos_best_g), output_dictionary


    # optimization function 1
    def evalutate_quad_opt(self, individual):
        x = np.array(individual, dtype=float)
        value = np.linalg.norm(np.matmul(self.A, x) - self.b, 2)
        return value

