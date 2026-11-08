''' Particle Swarm Optimisation with informants, following Algorithm 39
(Luke, Essentials of Metaheuristics, Section 3.5).

Comments of the format "Line N" refer to the line numbers in Algorithm 39.
'''

import numpy as np

class Particle:
    """One particle in the swarm: a candidate solution moving through the search space."""

    def __init__(self, dimensions, lower, upper):
        # Line 9: new random particle x with a random initial velocity v
        self.position = np.random.uniform(lower,upper,size=dimensions)
        self.velocity = np.random.uniform(-0.1,0.1,size=dimensions)
        
        self.fitness = None  # fitness of the current position
        self.best_position = None # x* (Line 17) : previous fittest position of this particle
        self.best_fitness = -np.inf # fitness of x* . Starts with -infinity so that the first evaluation of fitness will always be better than this.
        
        self.informants = [] # informants of this particle (Line 18), set later by PSO
      
    def assess_fitness(self,fitness_function):
        # Line 13: AssessFitness(x)
        self.fitness = fitness_function(self.position)
        
        if self.fitness > self.best_fitness: 
            self.best_fitness = self.fitness
            self.best_position = self.position.copy() 
    
    def move(self,epsilon):
        #  Line 26 : x <- x + epsilon * v
        self.position = self.position + epsilon * self.velocity 
    
    def informants_best_position(self):
        # Line 18 : x+ <- previous fittest location of the informants of x (including x itself)
        best_informant = self.informants[0] #assume the first informant is best 
        # iterate through every informant to see which one has the best fitness and return its best position
        for informant in self.informants:
            if informant.best_fitness > best_informant.best_fitness:
                best_informant = informant 
        return best_informant.best_position
        
    def update_velocity(self, alpha, beta, gamma, delta, global_best_position):
        x_star = self.best_position                 # Line 17: personal best (x*)
        x_plus = self.informants_best_position()    # Line 18: informants' best (x+)
        x_exclam = global_best_position             # Line 19: global best (x!)
        
        for i in range(len(self.position)):         #Line 20: for each dimension i do
            b = np.random.uniform(0.0, beta)        #Line 21
            c = np.random.uniform(0.0, gamma)       #Line 22
            d = np.random.uniform(0.0, delta)       #Line 23
            self.velocity[i] = (alpha * self.velocity[i] +  #Line 24
                                b*(x_star[i]-self.position[i]) + 
                                c*(x_plus[i] - self.position[i])+
                                d*(x_exclam[i] - self.position[i])
                                )
