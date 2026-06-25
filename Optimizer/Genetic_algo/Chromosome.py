class Chromosome:
    def __init__(self,genetic_code):
        self.genetic_code = genetic_code
        self.fitness = None
    
    def __repr__(self):
        return f"{self.genetic_code}"

    # again a bad design descision evaluate should not concern with weather to 
    # change internal state or not that's why its becoming verbose
    # seperate responsibility one function just evaluates fitness other update its value
    # less verbose
    
    # Method 1
    # def evaluate_fitness(self, change_current_fitness_score, fitness_metric):
    #     output = {value : fitness_metric(self), fitness_metric : fitness_metric}
    #     if(change_current_fitness_score):
    #         self.fitness = output

    #     return output

    # Method 2
    def evaluate_fitness(self, fitness_metric):
        output = fitness_metric(self)
        return output

    def update_fitness(self, fitness_metric):
        # self.fitness = fitness_metric(self) bad design two places where fitness_metric function getting called
        self.fitness = self.evaluate_fitness(fitness_metric)

