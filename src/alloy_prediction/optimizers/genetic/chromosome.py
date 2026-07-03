class Chromosome:
    def __init__(self, genetic_code):
        self.genetic_code = genetic_code
        self._fitness = None

    @property
    def composition(self):
        return self.genetic_code

    @property
    def fitness(self):
        return self._fitness
    
    @fitness.setter
    def fitness(self, new_fitness):
        self._fitness = new_fitness

    def __repr__(self):
        return f"{self.composition}"
    


    # chromosome should not know about evaluation of fitness
    # only genetic algo calculator must know about it
    # def evaluate_fitness(self, predictor, objective):
    #     property_value = predictor(self.composition)
    #     return objective(property_value)

    # def update_fitness(self, predictor, objective):
    #     self.fitness = self.evaluate_fitness(predictor, objective)
    #     return self.fitness