import numpy as np
import matplotlib.pyplot as plt
import random


def objectif_function(x):    
    return 5 + 0.5 * np.sin(2 * x) + 0.3 * (x - 2)**2




class GeneticAlgorithm:
    def __init__(self, population_size=20, mutation_rate=0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.bounds = (0, 5)  # x range [0,5]
        

    def initialize_population(self, initial_solutions):
        # Include the base solutions in initial population
        population = list(initial_solutions)
        # Add random solutions to reach population_size
        while len(population) < self.population_size:
            population.append(random.uniform(self.bounds[0], self.bounds[1]))
        return population
    




    def fitness(self, x):
        return -objectif_function(x)  # Negative because we're finding minimum
    

    def select_parents(self, population, fitness_scores):
        # Tournament selection
        tournament_size = 3
        parents = []
        for _ in range(2):
            tournament = random.sample(list(enumerate(population)), tournament_size)
            winner = max(tournament, key=lambda x: fitness_scores[x[0]])
            parents.append(winner[1])
        return parents
    
    def crossover(self, parent1, parent2):
        # Arithmetic crossover
        alpha = random.random()
        child = alpha * parent1 + (1 - alpha) * parent2
        return max(min(child, self.bounds[1]), self.bounds[0])
    
    def mutate(self, x):
        if random.random() < self.mutation_rate:
            delta = random.gauss(0, 0.5)
            x += delta
            x = max(min(x, self.bounds[1]), self.bounds[0])
        return x

        
    
    def evolve(self, generations=50, initial_solutions=None):
        if initial_solutions is None:
            initial_solutions = []
            
        population = self.initialize_population(initial_solutions)
        best_solution = None
        best_fitness = float('-inf')
        history = []
        
        for gen in range(generations):
            fitness_scores = [self.fitness(x) for x in population]
            
            # Track best solution
            gen_best_idx = max(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
            if fitness_scores[gen_best_idx] > best_fitness:
                best_fitness = fitness_scores[gen_best_idx]
                best_solution = population[gen_best_idx]
            
            history.append(-best_fitness)  # Convert back to minimization
            
            # Create new population
            new_population = []
            elite = population[gen_best_idx]  # Keep best solution
            new_population.append(elite)
            
            while len(new_population) < self.population_size:
                parents = self.select_parents(population, fitness_scores)
                child = self.crossover(parents[0], parents[1])
                child = self.mutate(child)
                new_population.append(child)
                
            population = new_population
            
        return best_solution, history




# Initial base solutions
base_solutions = [0.5, 2.2, 4.0, 4.2]
base_values = [objectif_function(x) for x in base_solutions]




# Create visualization
plt.figure(figsize=(12, 8))

# Plot the function
x = np.linspace(0, 5, 500)
y = objectif_function(x)

plt.plot(x, y, 'b-', label='f(x)')

# Plot base solutions
plt.plot(base_solutions, base_values, 'ro', label='Base Solutions', markersize=10)



# Run genetic algorithm
ga = GeneticAlgorithm(population_size=20, mutation_rate=0.1)
best_solution, history = ga.evolve(generations=50, initial_solutions=base_solutions)
# Plot the best solution found
best_value = objectif_function(best_solution)


plt.plot(best_solution, best_value, 'g*', label='GA Best Solution', markersize=15)

plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Function Optimization using Genetic Algorithm')

plt.legend()

# Add axis
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)

# Print results
print("\nBase Solutions:")
for x, fx in zip(base_solutions, base_values):
    print(f"f({x:.2f}) = {fx:.3f}")

print("\nGenetic Algorithm Result:")
print(f"Best solution found at x = {best_solution:.3f}")
print(f"Minimum value found: f(x) = {best_value:.3f}")

# Plot convergence history
plt.figure(figsize=(10, 4))
plt.plot(history, 'b-')
plt.grid(True)
plt.xlabel('Generation')
plt.ylabel('Best Fitness Value')
plt.title('Genetic Algorithm Convergence History')

plt.show()