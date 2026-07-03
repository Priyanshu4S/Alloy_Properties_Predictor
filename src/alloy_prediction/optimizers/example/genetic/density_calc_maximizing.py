import random 
from copy import deepcopy
import pandas as pd
from alloy_prediction.optimizers.genetic.chromosome import Chromosome
from alloy_prediction.optimizers.genetic.population import Population
from alloy_prediction.optimizers.genetic.genetic_algorithm import GeneticAlgorithm

from alloy_prediction.data.hea_data_loader import HEADataLoader
from alloy_prediction.models.svm_regression import (train_svm_regression_predictor,)
from alloy_prediction.models.neural_network import (train_neural_network_predictor,)


loader = HEADataLoader(
    csv_path="src/alloy_prediction/data/data_files/HEA Phase DataSet v1d.csv",
    target="Density_calc",
    excluded_columns=[
        "Alloy",
        "Alloy ID",
        "References",
        "Num_of_Elem",
        "dHmix",
        "dSmix",
        "dGmix",
        "Tm",
        "n.Para",
        "Atom.Size.Diff",
        "Elect.Diff",
        "VEC",
        "Sythesis_Route",
        "Hot-Cold_Working",
        "Homogenization_Temp",
        "Homogenization_Time",
        "Annealing_Temp",
        "Annealing_Time_(min)",
        "Quenching",
        "HPR",
        "Microstructure_",
        "Multiphase",
        "IM_Structure",
        "Microstructure",
        "Phases",
        "dHmix",
        "Unnamed: 51",
        "Unnamed: 52",
    ],
)

loader.prepare()

predictor_1, score_1 = train_svm_regression_predictor(loader)

print(f"Predictor R² SVM = {score_1:.3f}")

predictor_2, score_2 = train_neural_network_predictor(loader)

print(f"Predictor R² N_NET = {score_2:.3f}")

feature_names = loader.get_feature_names()

def predictor_svm_wrapper(composition):
    row = {feature: composition.get(feature, 0.0) for feature in feature_names}
    x = pd.DataFrame([row])
    # print(predictor.predict(x))
    return predictor_1.predict(x)[0]

def predictor_n_net_wrapper(composition):
    row = {feature: composition.get(feature, 0.0) for feature in feature_names}
    x = pd.DataFrame([row])
    # print(predictor.predict(x))
    return predictor_2.predict(x)[0]

ELEMENTS = [
    "Al", "Co", "Cr", "Fe", "Ni",
    "Cu", "Mn", "Ti", "V", "Nb",
    "Mo", "Zr", "Hf", "Ta", "W",
    "C", "Mg", "Zn", "Si", "Re",
    "N", "Sc", "Li", "Sn", "Be",
]

POPULATION_SIZE = 10

initial_population = Population(
    chromosomes=[],
    generation=0,
)

for _ in range(POPULATION_SIZE):

    # Choose between 3 and 8 alloying elements
    selected_elements = random.sample(ELEMENTS, random.randint(3, 8))

    # Generate random percentages for the selected elements
    values = [random.random() for _ in selected_elements]
    total = sum(values)

    # Start with every element absent
    genetic_code = {element: 0.0 for element in ELEMENTS}

    # Assign normalized percentages
    for element, value in zip(selected_elements, values):
        genetic_code[element] = value / total

    chromosome = Chromosome(genetic_code)

    initial_population.add_individual(chromosome)

initial_population_1 = deepcopy(initial_population)
initial_population_2 = deepcopy(initial_population)

def crossover_strategy(parent_1: Chromosome, parent_2: Chromosome) -> tuple[Chromosome, Chromosome]:
    child_1_code = {}
    child_2_code = {}

    for element in parent_1.composition:
        if random.random() < 0.5:
            child_1_code[element] = parent_1.composition[element]
            child_2_code[element] = parent_2.composition[element]
        else:
            child_1_code[element] = parent_2.composition[element]
            child_2_code[element] = parent_1.composition[element]
    # Normalize
    total = sum(child_1_code.values())
    if total > 0:
        child_1_code = {
            e: value / total
            for e, value in child_1_code.items()
        }

    total = sum(child_2_code.values())
    if total > 0:
        child_2_code = {
            e: value / total
            for e, value in child_2_code.items()
        }

    return Chromosome(child_1_code),Chromosome(child_2_code)
    
def mutating_strategy(
    chromosome,
    mutation_strength=0.1,
):

    element = random.choice(list(chromosome.composition.keys()))

    chromosome.composition[element] += (random.uniform(-mutation_strength,mutation_strength))

    chromosome.composition[element] = max(chromosome.composition[element], 0.0,)

    total = sum(chromosome.composition.values())

    if total > 0:
        for key in chromosome.composition:
            chromosome.composition[key] /= total

    chromosome.fitness = None

def selecting_strategy(population,tournament_size=3,):

    def tournament():
        contestants = random.sample(population.chromosomes,tournament_size,)

        return max(contestants,key=lambda chromosome: chromosome.fitness,)

    return (tournament(),tournament(),)

def objective_function(property_value):
    return property_value

def continue_generating(current_population,next_population,):
    return len(next_population) < len(current_population)

def make_stagnation_stopper(patience=20):

    counter = 0

    def stopping(previous_best,
                 current_best):

        nonlocal counter

        if current_best > previous_best:
            counter = 0
        else:
            counter += 1

        return counter >= patience

    return stopping

optimization_stopping_function = make_stagnation_stopper(20)

ga_1 = GeneticAlgorithm(continue_generating,crossover_strategy,mutating_strategy,objective_function,optimization_stopping_function,initial_population_1,predictor_svm_wrapper,selecting_strategy)
history_1 = ga_1.optimize()
# print(history_1)

ga_2 = GeneticAlgorithm(continue_generating,crossover_strategy,mutating_strategy,objective_function,optimization_stopping_function,initial_population_2,predictor_n_net_wrapper,selecting_strategy)
history_2 = ga_2.optimize()
# print(history_2)