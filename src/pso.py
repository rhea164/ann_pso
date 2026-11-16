import numpy as np
from src.particle import Particle  # package import so `python -m ...` from the project root works

class PSO:
    """The swarm and main loop of Algorithm 39"""
    
    def __init__ (self, fitness_function, dimensions, 
                  swarmsize=30,    #Line 1: desired swarm size
                  alpha=0.73,        #Line 2: proportion of velocity retained
                  beta=1.5,         #Line 3: proportion of personal best retained
                  gamma=1.0,        #Line 4: proportion of informants' best retained
                  delta=0.5,        #Line 5: proportion of global best retained
                  epsilon=1.0,      #Line 6: jump size of a particle
                  num_informants=3, #Number of informants per particle (not counting itself)
                  max_iterations=100, #Line 27: until run out of time
                  target_fitness=None, #Line 27: Best is the ideal solution
                  lower=-1.0,         #range for random initial positions
                  upper=1.0,
                  boundary_method="clamp" #method for handling particles that go out of bounds
                  ):
        self.fitness_function = fitness_function
        self.dimensions = dimensions
        self.swarmsize = swarmsize
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.num_informants = num_informants
        self.max_iterations = max_iterations
        self.target_fitness = target_fitness
        self.lower = lower
        self.upper = upper
        self.boundary_method = boundary_method
        
        self.swarm=[]
        self.best_position = None # Line 10 : Best <- Nothing
        self.best_fitness = -np.inf 
        self.history =[] #best fitness after each iteration, for convergence plots
        
        if num_informants >= swarmsize:
            raise ValueError("num_informants must be less than swarm_size")
        
    def initialise_swarm(self):
        self.swarm =[] #Line 7: P <-{}
        for _ in range(self.swarmsize):  #LINE 8-9 : for swarmsize times, add a new random particle with random velocity
            particle = Particle(self.dimensions, self.lower, self.upper)
            self.swarm.append(particle)
        self.set_informants() #Line 18 : set up the informants for each particle in the swarm
    
    def set_informants(self):
        # Line 18 needs each particle to have a list of informants. This method sets them up.
        # each particle gets itself + (num_informants) randomly chosen other particles as informants
        for i,particle in enumerate(self.swarm):
            others = [j for j in range(self.swarmsize) if j !=i] # all other particles except the current one
            chosen = np.random.choice(others, size=self.num_informants, replace=False) # choosing k random informants from all the other particles
            particle.informants = [particle] + [self.swarm[j] for j in chosen] # the informants of the current particle are : the current particle + the randomly chosen informants from the other particles
    
    def run(self):  #Line 7-28 : initialising and the main loop
        self.initialise_swarm() #Line 7-9: initialise the swarm with random particles and set their informants
        
        # setting best position again because init only runs once and run can be called multiple times 
        self.best_position = None # Line 10 : Best <- Nothing
        self.best_fitness = -np.inf 
        self.history =[] 
        self.escapes = 0
        
        for iteration in range(self.max_iterations): #Line 11: running the iterations. repeat until run out of time (Line 27)
            
            for particle in self.swarm: #Line 12
                particle.assess_fitness(self.fitness_function) #Line 13: AssessFitness(x)
                if self.best_position is None or particle.fitness > self.best_fitness: #Line 14: if Best is Nothing or Fitness(x) > Fitness(Best)
                    self.best_position = particle.position.copy() #Line 15: Best <- x
                    self.best_fitness = particle.fitness #Line 15
                    
            self.history.append(self.best_fitness) #record the best fitness after each iteration, for convergence plots
            
            for particle in self.swarm: #Line 16
                particle.update_velocity(self.alpha,self.beta,self.gamma,self.delta,self.best_position) #Line 17-24: update the velocity of each particle
                
            for particle in self.swarm: #Line 25
                particle.move(self.epsilon) #Line 26: move the particle to its new position
                if np.any((particle.position < self.lower) | (particle.position > self.upper)):
                    self.escapes += 1 #count how many times particles escape the search space
                particle.apply_boundary_handling(self.lower, self.upper, self.boundary_method) #EXTENSION: boundary handling
                
                
            if self.target_fitness is not None and self.best_fitness >= self.target_fitness:
                break #Line 27: until Best is the ideal solution
            
        return self.best_position, self.best_fitness #Line 28: return Best

if __name__== "__main__":
    # testing boundary handling techniques 
    def peak_near_wall(x):
        """A simple fitness function with a peak near the upper boundary."""
        return -np.sum((x - 4.5)**2)
    
    for method in [None, "clamp", "random_reset"]:
        pso = PSO(peak_near_wall, dimensions=2, max_iterations=100,
                  lower=-5, upper=5.0, boundary_method=method)
        best_vector, best_fitness = pso.run()
        inside = all(np.all((p.position >= pso.lower) & (p.position <= pso.upper)) for p in pso.swarm)
        print (f"Method: {method} | Best fitness: {best_fitness:} | All particles inside bounds? {inside} | Escapes: {pso.escapes}")
        