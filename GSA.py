import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from convex_quadratic_opt import generate_input as gi
np.set_printoptions(linewidth=np.inf)

def to_string():
    return "GSA"
   
class Algorithm:
    
    def __init__(self, **args):
        self.A = None
        self.b = None
        self.A, self.b = gi(args["k"], args["size"], args["debug"])
        self.debug = args["debug"] != -1

        print(self.A)
        print(self.b)
        self.G =  None
        self.max_iter  = args["number_generations"]
        self.pop_size  = args["pop_size"]
        self.dimension = args["size"]
        
        if args["problems"] == 1:
          self.cost_func = self.evalutate_quad_opt
        else:
          self.cost_func = self.evalutate_quad_opt
       
        self._current_iter = 0
        self.best_so_far_iteration = None
        self.best_so_far = None
        self.worse_so_far = None
        
        
        self.fig = plt.figure()
        self.ax = plt.axes()
        self.arrows = []
        self.line, = self.ax.plot([], [], 'o', color='black')
        self.history_loc = []
        self.locations = np.array([[ random.randint(-30, 30) for i in range(self.dimension) ] for i in range(self.pop_size)])        
        self.history_loc.append(self.locations)
        self.history_V = []        
        
        self.V = np.random.rand(self.pop_size, self.dimension)
        self.history_V.append(self.V)
        self.f = np.full((self.pop_size, self.dimension), None)  # None  # Will become a list inside cal_f
        self.accel = np.full((self.pop_size, self.dimension), None)
        self.inertia = np.full((self.pop_size, 1), None)
        self.mass_matrix = np.full((self.pop_size, 1), None)
        self.cost_matrix = np.full((self.pop_size, 1), None)
        
    # optimization function 1
    def evalutate_quad_opt(self, individual):
        x = np.array(individual, dtype=float)
        value = 0.5 * np.matmul(np.matmul(x.T, self.A), x) - np.matmul(self.b.T, x)
        return value

    # Evaluate a single x (x_i)
    def evaluate(self, args):
        return self.cost_func(args) 

    # Generate the cost of all particles
    def gen_cost_matrix(self):
        if self.debug :
            print("\tloc: \n{}".format(self.locations))
        for i, x in enumerate(self.locations):
            self.cost_matrix[i] = self.evaluate(x)
        if self.debug :
            print("\tcmx: \n{}".format(self.cost_matrix))
        self.cal_inertia()

    def cal_inertia(self):
        best = np.min(self.cost_matrix)
        worst = np.max(self.cost_matrix)
        self.inertia = (self.cost_matrix - worst) / best - worst
        if self.debug:
            print("\tine: \n{}".format(self.inertia))
        self.cal_mass()
            
    def cal_mass(self):
        self.mass_matrix = self.inertia / np.sum(self.inertia)
        if self.debug :
            print("\tmas: \n{}".format(self.mass_matrix))
        self.update_grav()


    def update_grav(self):
        self.G = 10 *  np.exp(-1 * (self._current_iter / self.max_iter) )
        self.cal_force()
        
    def cal_force(self):
        costs = self.cost_matrix.copy()
        costs.sort(axis=0)
        
        for i in range(self.pop_size):
            f = None
            for j in range(costs.size):
                dividend = float(self.mass_matrix[i] * self.mass_matrix[j])
                divisor = np.sqrt(np.sum((self.locations[i] - self.locations[j]) ** 2)) + np.finfo('float32').eps
                if f is None:
                    f = self.G * (dividend / divisor) * (self.locations[j] - self.locations[i])
                else:
                    f = f + self.G * (dividend / divisor) * (self.locations[j] - self.locations[i])

            self.f[i] = f * np.random.uniform(0,1)
            
        if self.debug :
            print("\tfce: \n{}".format(self.f))
        
        self.cal_accel()

    def cal_accel(self):
        for i in range(self.pop_size):
            self.accel[i] = self.f[i] / self.mass_matrix[i]
        self.cal_vel()

    def cal_vel(self):
        self.V = (np.random.uniform(0, 1) * self.V) + self.accel
        self.history_V.append(self.V)
        if self.debug :
            print("\tvel: \n{}".format(self.V))
        self.update_locations()
        
        
    def update_locations(self):
        self.locations = self.locations + self.V
        if self.debug :
            print("\tnlc: \n{}".format(self.locations))
        self.update_best_so_far()

    # set for minimization
    def update_best_so_far(self):
        
        best = np.min(self.cost_matrix)
        index = np.where(self.cost_matrix == best)[0][0]
        if self.best_so_far is None:
            self.best_so_far = self.locations[index]
            self.best_so_far_iteration = self._current_iter
            if self.debug :
                print("new best: {} @ {}".format(self.best_so_far, self.evaluate(self.best_so_far)))
        
        if self.evaluate(self.best_so_far) > self.evaluate(self.locations[index]):
            self.best_so_far_iteration = self._current_iter
            self.best_so_far = self.locations[index]
            if self.debug :
                print("------: {} > {}".format(self.evaluate(self.best_so_far), self.evaluate(self.locations[index])))
                print("new best: {} @ {}".format(self.best_so_far, self.evaluate(self.best_so_far)))
            
            
        worse = np.max(self.cost_matrix)
        index = np.where(self.cost_matrix == worse)[0][0]
        if self.worse_so_far is None :
            self.worse_so_far = self.locations[index]
            if self.debug :
                print("new worse: {} @ {}".format(self.worse_so_far, self.evaluate(self.worse_so_far)))
         
        if self.evaluate(self.worse_so_far) < self.evaluate(self.locations[index]):
            if self.debug :
                print("------: {} < {}".format(self.evaluate(self.worse_so_far), self.evaluate(self.locations[index])))
            self.worse_so_far = self.locations[index]
            if self.debug :
                print("new worse: {} @ {}".format(self.worse_so_far, self.evaluate(self.worse_so_far)))


    def init_animation(self):
        self.line.set_data([],[])
        return self.line, 

    def animate(self, time):
        print(time)
        x = []
        y =[]
        for i in self.history_loc[time]:
            x.append(i[0])
            y.append(i[1])
        
        
        #if len(self.arrows) > 2:
        #    for i in self.arrows:
        #        i.remove()
        #self.arrows = []
        #dx = []
        #dy =[]
        #for i in self.history_V[time]:
        #    dx.append(i[0])
        #    dy.append(i[1])
        #for x0,y0,dx0,dy0 in zip(x,y,dx,dy):
        #    new_arrow = self.ax.arrow(x0,y0,dx0,dy0, head_width= .5, head_length=.5)
        #   self.arrows.append(new_arrow)
        #self.arrows.append(self.ax.plot(self.best_so_far[0], self.best_so_far[1], 'x', 'red'))
        
        self.line.set_data(x,y)
        
        return self.line, 
           
    def run(self):
        
        avg = []
        min_results = [] 
        max_results = []
        avg = []
        std = []        
        while self._current_iter <= self.max_iter:
            self.gen_cost_matrix()
            self.history_loc.append(self.locations)
            self.history_V.append(self.V)
            
        
            iterave = sum(self.cost_matrix) / len(self.cost_matrix)
            
            avg.append(iterave[0])         
            
            min_results.append(min(self.cost_matrix.flatten()))
            max_results.append(max(self.cost_matrix.flatten()))
            std.append(np.std(min_results))
                       

            self._current_iter += 1
            
            print("{} ---bsf: {} | cost: {}".format(self._current_iter -1 , self.best_so_far, self.evaluate(self.best_so_far)))
            print("===="*20)

        self.history_loc.append(self.locations)
        self.history_V.append(self.V)
        self.ax.axis([-20,20,-20,20])
        anim = FuncAnimation(self.fig, self.animate, init_func=self.init_animation, frames=self._current_iter+2, interval=500, blit=True)
        x, y = np.meshgrid(np.linspace(-20, 20, 500),np.linspace(-20, 20, 500))
        
        
        for i,j in zip(x,y) :        
             plt.contourf(x, y, self.evaluate([i,j]))
             
        plt.colorbar()
        anim.save('animate.gif', writer='imagemagick')
        print("\tBest individual seen fitness value: {}".format(self.evaluate(self.best_so_far )))
        print("\tBest individual seen generation apperared in: {}".format(self.best_so_far_iteration ))

        output_dictionary = {"iterations": [i for i in range(self.max_iter)], "avg": avg, "min": min_results, "max": max_results, "std": std}
        return self.best_so_far, self.evaluate(self.best_so_far), output_dictionary

