from alloy_prediction.optimizers.genetic.chromosome import Chromosome


class Population:
    def __init__(self, chromosomes: list[Chromosome], generation):
        self.chromosomes = chromosomes
        self._generation = generation

    @property
    def generation(self):
        return self._generation

    @generation.setter
    def generation(self, generation_count):
        self._generation = generation_count

    def __iter__(self):
        return iter(self.chromosomes)

    def __len__(self):
        return len(self.chromosomes)

    def __getitem__(self, index):
        return self.chromosomes[index]

    def add_individual(self, chromosome):
        self.chromosomes.append(chromosome)

    def best_individual(self) -> Chromosome:
        return max(
            self.chromosomes,
            key=lambda chromosome: chromosome.fitness,
        )

    @property
    def best_fitness(self):
        return self.best_individual().fitness

    @property
    def average_fitness(self):
        return sum(c.fitness for c in self.chromosomes) / len(self.chromosomes)