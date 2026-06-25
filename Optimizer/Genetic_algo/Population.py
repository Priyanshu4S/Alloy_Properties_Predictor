from Chromosome import Chromosome
class Population:
    def __init__(self,chromosomes: list[Chromosome]):
        self.chromosomes = chromosomes
    
    def __iter__(self):
        return iter(self.chromosomes)

    def add_individual(self, chromosome):
        self.chromosomes.append(chromosome)

    def evaluate(self, fitness_metric):
        fitness_scores = {}
        for individual in self:
            fitness_scores[individual] = individual.evaluate_fitness(fitness_metric)

        return fitness_scores
