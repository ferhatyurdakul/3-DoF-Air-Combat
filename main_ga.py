import numpy as np
import random
from Aircraft_3DoF import Aircraft
from AirCombatEnv_GA import F16Environment

class GeneticAlgorithmDogfight:
    def __init__(self, population_size=100, generations=1_000, mutation_rate=0.2):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.population = []
        self.env = F16Environment()

    def initialize_population(self):
        """
        Initialize the population with random headings for both aircraft.
        """
        self.population = [
            (random.uniform(0, 2 * np.pi), random.uniform(0, 2 * np.pi))
            for _ in range(self.population_size)
        ]

    def fitness(self, heading1, heading2):
        """
        Evaluate the fitness of the scenario by simulating the dogfight.
        """
        # Reset the environment with initial headings
        self.env.reset(heading1, heading2)

        max_steps = 15
        turning_rate = np.pi / 36  # Constant turning rate (5 degrees per step)

        for step in range(max_steps):
            # Aircraft 1 turns towards Aircraft 2
            target_heading1 = np.arctan2(
                self.env.aircraft2.y - self.env.aircraft1.y,
                self.env.aircraft2.x - self.env.aircraft1.x
            )
            heading_diff1 = target_heading1 - self.env.aircraft1.psi
            self.env.aircraft1.psi += np.clip(heading_diff1, -turning_rate, turning_rate)

            # Aircraft 2 turns towards Aircraft 1
            target_heading2 = np.arctan2(
                self.env.aircraft1.y - self.env.aircraft2.y,
                self.env.aircraft1.x - self.env.aircraft2.x
            )
            heading_diff2 = target_heading2 - self.env.aircraft2.psi
            self.env.aircraft2.psi += np.clip(heading_diff2, -turning_rate, turning_rate)

            # Step in the environment with no external actions
            action = np.zeros(6)
            _, fitness, done, _ = self.env.step(action)

        return fitness

    def select_parents(self):
        """
        Select parents using tournament selection.
        """
        tournament_size = 3
        parents = random.sample(self.population, tournament_size)
        fitness_scores = [self.fitness(parent[0], parent[1]) for parent in list(parents)]
        return parents[np.argmax(fitness_scores)]

    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parents to produce an offspring.
        """
        alpha = random.uniform(0, 1)
        child1 = alpha * parent1[0] + (1 - alpha) * parent2[0]
        child2 = alpha * parent1[1] + (1 - alpha) * parent2[1]
        return (child1, child2)

    def mutate(self, individual):
        """
        Apply mutation to an individual's headings.
        """
        if random.uniform(0, 1) < self.mutation_rate:
            individual = (
                individual[0] + random.uniform(-0.1, 0.1),
                individual[1] + random.uniform(-0.1, 0.1),
            )
        return individual

    def run(self):
        """
        Run the genetic algorithm to find the best initial headings.
        """
        self.initialize_population()
        for generation in range(self.generations):
            new_population = []
            for _ in range(self.population_size // 2):
                # Select parents
                parent1 = self.select_parents()
                parent2 = self.select_parents()

                # Perform crossover
                child1 = self.crossover(parent1, parent2)
                child2 = self.crossover(parent2, parent1)

                # Mutate offspring
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                new_population.extend([child1, child2])

            # Replace the old population
            self.population = new_population

            # Evaluate and print the best fitness in this generation
            fitness_scores = [self.fitness(*individual) for individual in self.population]
            best_fitness = max(fitness_scores)
            best_individual = self.population[np.argmax(fitness_scores)]
            print(f"Generation {generation + 1}: Best Fitness = {best_fitness}")

        # Return the best individual
        return best_individual

if __name__ == "__main__":
    ga = GeneticAlgorithmDogfight()
    best_headings = ga.run()
    print(f"Best Initial Headings: Aircraft 1 = {best_headings[0]}, Aircraft 2 = {best_headings[1]}")
