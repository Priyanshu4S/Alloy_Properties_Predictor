import copy
from Population import Popluation


class Genetic_algo:
    def __init__(self,crossover,fitness_metric,mutator,population: Popluation,selector_function):

        self.population = population
        self.crossover = crossover
        self.fitness_metric = fitness_metric
        self.mutator = mutator
        self.selector_function = selector_function
               

    def next_generation(self,stopping_function):
        # needed for method 2
        fitness_scores = self.population.evaluate(self.fitness_metric,0)

        next_population = Population([])
        # stopping_function will output 1 uptil a desired condition is filled for
        # given current population and evolving next population
        while stopping_function(self.population,next_population):
            # design pattern 1
            # parent_1, parent_2 = self.selector_function(
            #     self.population,
            #     self.fitness_evaluator
            # ) 

            # bad design pattern selector need to remember the 
            # evaluation metric and need to call it explicitly 
            # inside its operation need to do seperation of 
            # concern wants just fitness scores to work with not
            # the full method how fitness scores are calculated 
            
            # Method 2

            parent_1, parent_2 = self.selector_function(
                self.population,
                fitness_scores,
            )

            child_1, child_2 = self.crossover(parent_1,parent_2)

            self.mutator(child_1)
            self.mutator(child_2)
            
            next_population.add_individual(child_1)
            next_population.add_individual(child_2)

        return next_population

    def give_nth_generation(self, n, stopping_function):

        for i in range(n):
            self.population = self.next_generation(stopping_function)

        return self.population

        
