
import numpy as np
from convex_quadratic_opt import generate_input as gi

def to_string():
    return "GSA"

A , b = None, None

# optimization function    
def evalutate_quad_opt(individual):
    global A
    global b
    x = np.array(individual, dtype=float)
    value = 0.5 * np.matmul(np.matmul(x.T, A), x) - np.matmul(b.T, x)
    return 1/value

       
class Algorithm:
    def __init__(self, **args):
        global A
        global b
        A, b = gi(args["k"], args["size"], args["debug"])
        
        self.alpha = 0.1
        self.G = 0.9
        self.max_iter  = args["number_generations"]
        self.pop_size  = args["pop_size"]
        self.dimension = args["size"]
        
        if args["problems"] == 1:
          self.cost_func = evalutate_quad_opt
        else:
          self.cost_func = evalutate_quad_opt
         
        self.best_so_far = None
        self.worse_so_far = None
        self.write_out = ""
        self.X = np.random.rand(self.pop_size, self.dimension)
        self.V = np.random.rand(self.pop_size, self.dimension)
        self.f = np.full((self.pop_size, self.dimension), None)  # None  # Will become a list inside cal_f
        self.a = np.full((self.pop_size, self.dimension), None)
        self.q = np.full((self.pop_size, 1), None)
        self.M = np.full((self.pop_size, 1), None)
        self.cost_matrix = np.full((self.pop_size, 1), None)

    # Evaluate a single x (x_i)
    def evaluate(self, args):
        return self.cost_func(args)

    # Generate the cost of all particles
    def gen_cost_matrix(self):
        #print("x: {}".format(self.X))
        for i, x in enumerate(self.X):
            self.cost_matrix[i] = self.evaluate(x)
        #print("cost matrix: {}".format(self.cost_matrix))

    def cal_q(self):

        best = np.min(self.cost_matrix)
        worst = np.max(self.cost_matrix)

        self.q = (self.cost_matrix - worst) / best - worst

    def cal_m(self):
        self.M = self.q / np.sum(self.q)

    def cal_f(self):
        costs = self.cost_matrix.copy()
        costs.sort(axis=0)
        costs = costs

        for i in range(self.pop_size):
            f = None
            for cost in costs:
                j = int(np.where(self.cost_matrix == cost)[0])

                dividend = float(self.M[i] * self.M[j])
                divisor = np.sqrt(np.sum((self.X[i] - self.X[j]) ** 2)) + np.finfo('float').eps
                if f is None:
                    f = self.G * (dividend / divisor) * (self.X[j] - self.X[i])
                else:
                    f = f + self.G * (dividend / divisor) * (self.X[j] - self.X[i])

            self.f[i] = np.random.uniform(0, 1) * f

    def cal_a(self):
        for i in range(self.pop_size):
            self.a[i] = self.f[i] / self.M[i]

    def cal_v(self):
        self.V = (np.random.uniform(0, 1) * self.V) + self.a

    def move(self):
        self.X = self.X + self.V

    def update_g(self, iteration):
        self.G = self.G * np.e ** (- self.alpha * (iteration / self.max_iter))

    def show_results(self):
        print('Best seen so far is located at:', self.best_so_far, 'Cost:', self.evaluate(self.best_so_far))


    # set for minimization
    def update_best_so_far(self):
        best = np.min(self.cost_matrix)
        index = int(np.where(self.cost_matrix == best)[0])
        if self.best_so_far is None or self.evaluate(self.best_so_far) > self.evaluate(self.X[index]):
            self.best_so_far = self.X[index]
            print("new best: {}".format(self.evaluate(self.best_so_far)))
            
        
        worse = np.max(self.cost_matrix)
        index = int(np.where(self.cost_matrix == worse)[0])
        self.worse_so_far = self.X[index]
        print("new worse: {}".format(self.evaluate(self.worse_so_far)))

    def run(self):
        _iter = 0
        avg = []
        min_results = [] # bsf
        max_results = []# wsf
        avg = []#ave
        std = []#[...]
        while _iter < self.max_iter:
            self.gen_cost_matrix()
            self.cal_q()
            self.cal_m()
            self.update_g(_iter)
            self.cal_f()
            self.cal_a()
            self.cal_v()
            self.move()
            self.update_best_so_far()

            iterave = sum(self.cost_matrix) / len(self.cost_matrix)
            avg.append(iterave)         
            
            min_results.append(self.evaluate(self.best_so_far))
            max_results.append(self.evaluate(self.worse_so_far))
            std.append(np.std(min_results))
            
            self.write_out += "itr: {} |avgcost: {} |bsf: {}|wsf: {}\n".format(iterave, _iter, min_results[-1], max_results[-1]) 
            
            
            print("itr: {} |avgcost: {} |bsf: {}".format(iterave, _iter, min_results[-1]) )
            _iter += 1
        
        output_dictionary = {"iterations": [i for i in range(1, self.max_iter+1)], "avg": avg, "min": min_results, "max": max_results, "std": std}
        return self.best_so_far, self.evaluate(self.best_so_far), output_dictionary

