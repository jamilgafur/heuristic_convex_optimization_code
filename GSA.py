import random
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

        self.alpha = args["alpha"]
        self.G = 100# random.uniform(0, 100) 
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
        self.write_out = ""
        
        self.locations = np.array([[ random.uniform(0, 100) for i in range(self.dimension) ] for i in range(self.pop_size)])
        
        #np.random.rand(self.pop_size, self.dimension)

        #np.array([[ np.random.randint(100) for i in range(0,self.dimension) ] for i in range(0, self.pop_size)])
        

        self.V = np.random.rand(self.pop_size, self.dimension)
        self.f = np.full((self.pop_size, self.dimension), None)  # None  # Will become a list inside cal_f
        self.accel = np.full((self.pop_size, self.dimension), None)
        self.inertia = np.full((self.pop_size, 1), None)
        self.mass_matrix = np.full((self.pop_size, 1), None)
        self.cost_matrix = np.full((self.pop_size, 1), None)
        
    # optimization function 1
    def evalutate_quad_opt(self, individual):
        x = np.array(individual, dtype=float)
        #value = 0.5 * np.matmul(np.matmul(x.T, self.A), x) - np.matmul(self.b.T, x)
        value = np.linalg.norm(np.matmul(self.A, x) - self.b, 2)
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
        if self.debug :
            print("\tine: \n{}".format(self.inertia))
        self.cal_mass()
            
    def cal_mass(self):
        self.mass_matrix = self.inertia / np.sum(self.inertia)
        if self.debug :
            print("\tmas: \n{}".format(self.mass_matrix))
        self.update_grav()


    def cal_force(self):
        costs = self.cost_matrix.copy()
        costs.sort(axis=0)
        
        for i in range(self.pop_size):
            f = None
            for cost in costs:
                j = int(np.where(self.cost_matrix == cost)[0])

                dividend = float(self.mass_matrix[i] * self.mass_matrix[j])
                divisor = np.sqrt(np.sum((self.locations[i] - self.locations[j]) ** 2)) + np.finfo('float32').eps
                if f is None:
                    f = self.G * (dividend / divisor) * (self.locations[j] - self.locations[i])
                else:
                    f = f + self.G * (dividend / divisor) * (self.locations[j] - self.locations[i])

            self.f[i] = f 
            
        if self.debug :
            print("\tfce: \n{}".format(self.f))
        
        self.cal_accel()

    def cal_accel(self):
        for i in range(self.pop_size):
            self.accel[i] = self.f[i] / self.mass_matrix[i]
        self.cal_vel()

    def cal_vel(self):
        self.V = (np.random.uniform(0, 1) * self.V) + self.accel
        if self.debug :
            print("\tvel: \n{}".format(self.V))
        self.update_locations()
        
    def update_grav(self):
        self.G = self.G *  (1 - (self._current_iter / self.max_iter) )
        self.cal_force()
        
    def update_locations(self):
        self.locations = self.locations + self.V
        if self.debug :
            print("\tnlc: \n{}".format(self.locations))
        self.update_best_so_far()

    # set for minimization
    def update_best_so_far(self):
        best = np.min(self.cost_matrix)
        index = int(np.where(self.cost_matrix == best)[0])
        
        if self._current_iter > 1:
        
            if self.debug :
                print("checking: {} @ {}".format(self.locations[index], self.evaluate(self.locations[index])))
                print("with: {} @ {}".format(self.best_so_far, self.evaluate(self.best_so_far)))
            
        
        if self.best_so_far is None:
            self.best_so_far = self.locations[index]
            self.best_so_far_iteration = self._current_iter
            if self.debug :
                print("new best: {} @ {}".format(self.best_so_far, self.evaluate(self.best_so_far)))
            
        if self.evaluate(self.best_so_far) > self.evaluate(self.locations[index]):
            if self.debug :
                print("------: {} > {}".format(self.evaluate(self.best_so_far), self.evaluate(self.locations[index])))
            self.best_so_far_iteration = self._current_iter
            self.best_so_far = self.locations[index]
            if self.debug :
                print("new best: {} @ {}".format(self.best_so_far, self.evaluate(self.best_so_far)))
            
        
        worse = np.max(self.cost_matrix)
        index = int(np.where(self.cost_matrix == worse)[0])
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

        
            
    def run(self):
        
        avg = []
        min_results = [] # bsf
        max_results = []# wsf
        avg = []#ave
        std = []#[...]
        while self._current_iter <= self.max_iter:
            self.gen_cost_matrix()
            
            iterave = sum(self.cost_matrix) / len(self.cost_matrix)
            
            avg.append(iterave)         
            
            iteration_best = int(np.where(self.cost_matrix == np.min(self.cost_matrix))[0])
            iteration_worse = int(np.where(self.cost_matrix == np.max(self.cost_matrix))[0])
            
            min_results.append(self.locations[iteration_best])
            max_results.append(self.locations[iteration_worse])
            std.append(np.std(min_results))
            
            self._current_iter += 1
            
            if self.debug:
                print("{} ---bsf: {} | cost: {}".format(self._current_iter -1 , self.best_so_far, self.evaluate(self.best_so_far )))
                print("===="*20)
        
        print("\tBest individual seen fitness value: {:0.3f}".format(self.evaluate(self.best_so_far )))
        print("\tBest individual seen generation apperared in: {}".format(self.best_so_far_iteration ))

        output_dictionary = {"iterations": [i for i in range(1, self.max_iter+1)], "avg": avg, "min": min_results, "max": max_results, "std": std}
        return self.best_so_far, self.evaluate(self.best_so_far), output_dictionary

