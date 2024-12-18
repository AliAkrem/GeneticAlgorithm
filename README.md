# Les Algorithmes Génétiques : Un Pilier des Algorithmes Évolutionnaires

Les algorithmes génétiques (AG) constituent une sous-catégorie majeure des algorithmes évolutionnaires. Inspirés par la génétique et les mécanismes d’évolution biologique, les AG utilisent des concepts tels que les chromosomes, les croisements et les mutations pour modéliser les solutions potentielles sous forme d’une population. Ces solutions évoluent au fil des générations, les plus aptes étant sélectionnées pour créer de nouvelles solutions. Les AG se distinguent par leur flexibilité et leur capacité à s’adapter à une grande variété de problèmes, allant de l’optimisation combinatoire à la recherche de solutions dans des environnements complexes. En tant qu’élément fondamental des AE, les algorithmes génétiques démontrent comment des processus biologiques peuvent être transformés en outils puissants pour résoudre des défis technologiques.

![AE](https://github.com/user-attachments/assets/a2f686d0-bf3a-441e-a396-072cc605caef)


---

## Classe GeneticAlgorithm

La classe principale GeneticAlgorithm encapsule toute la logique de l'algorithme génétique. Elle est initialisée avec deux paramètres principaux :

-  population_size  : Nombre d'individus dans la population (par défaut 20)
-  mutation_rate  : Probabilité de mutation pour chaque individu (par défaut 0.1)

L'espace de recherche est limité à l'intervalle [0, 5].

```py
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
```


## Initialisation de la Population

La méthode initialize_population  crée la population initiale en :

-  Intégrant des solutions de base fournies
-  Complétant avec des solutions aléatoires jusqu'à atteindre la taille de population désirée

## Évaluation de la Fitness

La méthode fitness  évalue la qualité de chaque solution. Elle inverse le signe de la fonction objectif car l'algorithme est conçu pour maximiser la fitness alors que nous cherchons à minimiser la fonction.

## Sélection des Parents

La méthode select_parents  utilise une sélection par tournoi :

-  Sélectionne aléatoirement 3 individus
-  Choisit le meilleur comme parent
-  Répète le processus pour obtenir deux parents

## Croisement

La méthode crossover  implémente un croisement arithmétique :

-  Combine deux parents avec un coefficient aléatoire
-  Garantit que l'enfant reste dans les bornes définies

## Mutation

La méthode mutate  applique une mutation gaussienne :

-  Probabilité de mutation définie par mutation\_rate
-  Perturbation suivant une distribution normale
-  Maintien des solutions dans les bornes

```py

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
```


## Évolution

La méthode evolve  orchestre le processus évolutif :

-  Initialise la population
-  Pour chaque génération : 
    -  Évalue la fitness de la population
    -  Conserve la meilleure solution (élitisme)
    -  Crée une nouvelle population par sélection, croisement et mutation
-  Suit l'historique de convergence


```py
   
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
```



## Fonction Objectif

La fonction objective_function(x) définit le problème d'optimisation à résoudre. Elle combine trois éléments :
-  Une constante (5)
-  Un terme sinusoïdal (0.5 * sin(2x))
-  Un terme quadratique (0.3 * (x-2)²)


Cette fonction crée un paysage d'optimisation avec plusieurs minimums locaux, ce qui en fait un bon cas test pour l'algorithme génétique.

```py
def objectif_function(x):    
    return 5 + 0.5 * np.sin(2 * x) + 0.3 * (x - 2)**2
```

![objectif_function](https://github.com/user-attachments/assets/543d5c5e-0d80-41b8-a595-058f22d20515)



## Utilisation et Paramétrage

Pour utiliser cet algorithme, 
Définir les solutions de base :

```py
# Initial base solutions
base_solutions = [0.5, 2.2, 4.0, 4.2]
base_values = [objectif_function(x) for x in base_solutions]
```

Créer une instance de l'algorithme :

```py
# Run genetic algorithm
ga = GeneticAlgorithm(population_size=20, mutation_rate=0.1)
best_solution, history = ga.evolve(generations=50, initial_solutions=base_solutions)
# Plot the best solution found
best_value = objectif_function(best_solution)
```



Les paramètres clés à ajuster sont :

-  La taille de la population : influence la diversité génétique
-  Le taux de mutation : contrôle l'exploration de l'espace
-  Le nombre de générations : détermine le temps d'optimisation

## Résultats et Interprétation


![final_result](https://github.com/user-attachments/assets/4c4c4e08-06da-43e4-942a-a2f7c6dea25c)


