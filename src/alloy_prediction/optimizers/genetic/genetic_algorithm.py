import copy
from alloy_prediction.optimizers.genetic.population import Population
from alloy_prediction.optimizers.base_optimizer import BaseOptimizer

class GeneticAlgorithm(BaseOptimizer):
    def __init__(self, continue_generating, crossover, mutator, objective, optimization_stopping_function, population: Population, predictor, selector_function):
        super().__init__(predictor, objective)

        self.population = population
        self.crossover = crossover
        self.mutator = mutator
        self.selector_function = selector_function
        self.continue_generating = continue_generating
        self.optimization_stopping_function = optimization_stopping_function

   
    def _evaluate_population(self, population : Population):
        for chromosome in population:
            property_value = self.predictor(chromosome.composition)
            chromosome.fitness = self.objective(property_value)
            # return objective(property_value)
            # chromosome.update_fitness(self.predictor,self.objective,)
   
    def _generate_next_population(self, population):
        # self._evaluate_population(population)

        next_population = Population(chromosomes=[],generation=population.generation + 1,)

        next_population.add_individual(copy.deepcopy(population.best_individual()))  

        while self.continue_generating(population,next_population):
            parent_1, parent_2 = self.selector_function(population,)

            child_1, child_2 = self.crossover(parent_1,parent_2)

            self.mutator(child_1)
            self.mutator(child_2)
            
            next_population.add_individual(child_1)
            next_population.add_individual(child_2)

        return next_population
    

    # internal state changing functions
    def next_generation(self):
        self.population = self._generate_next_population(self.population)
        return self.population
    
    def advance_n_generations(self, n):
        self._evaluate_population(self.population)
        for _ in range(n):
            self.next_generation()
            self._evaluate_population(self.population)
        return self.population

    def predict_nth_generation(self, n):
        output = copy.deepcopy(self.population)
        self._evaluate_population(output)
        for _ in range(n):
            output = self._generate_next_population(output)
            self._evaluate_population(output)
        return output


    def optimize(self):
        history = []
        
        self._evaluate_population(self.population)
        while True:
            # self._evaluate_population(self.population)
            obj = {
                "generation": self.population.generation,
                "best": self.population.best_fitness,
                "average": self.population.average_fitness,
                "best_individual" : copy.deepcopy(self.population.best_individual())
            }
            # print(obj,"\n")
            history.append(obj)

            previous_best = self.population.best_fitness
            self.next_generation()
            self._evaluate_population(self.population)

            if self.optimization_stopping_function(previous_best,self.population.best_fitness,):
                break

        return history
                    
                
