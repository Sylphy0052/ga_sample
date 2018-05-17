import random
from deap import base
from deap import creator
from deap import tools

# creator is a class factory
# takes 2 arguments
# a maximizing fitness is replaced for virtual weights attribute by (1.0,)
# that means to maximize a single objective
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# evaluation function is pretty simple
# need to count the number of ones in the individual
def evalOneMax(individual):
    return sum(individual),

def toolboxInitializer():
    # toolbox is to store functions with their arguments
    # toolbox contains two methods, register and unregister
    # a generation function and two initialization functions ia registered
    toolbox = base.Toolbox()
    # Attribute generator
    # The generator toolbox.attr_bool() when called, will draw a random integer between 0 and 1
    # attr_bool generator is made from the randint that takes two arguments a and b, with a <= n <= b
    # n is the returned integer
    toolbox.register("attr_bool", random.randint, 0, 1)
    # Structure initializers
    # individual method, initRepeat takes 3 arguments, a container class -individual is derived from a list-
    # function to fill the container and the number of times the function shall be repeated
    # individual method will thus return an individual initialized with what would be returned by 100 calls to the attr_bool method
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 100)
    # population method used the same paradigm, but we don't fix the number of individuals that it should contain
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # The Genetic Operators
    # Registering the operators and their default arguments in the toolbox is done as follow
    # register the goal / fitness function
    toolbox.register("evaluate", evalOneMax)
    # register the crossover operator
    toolbox.register("mate", tools.cxTwoPoint)
    # register a mutation operator with a probability to flip each attribute/gene of 0.05
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    # operator for selecting individuals for breeding the next generation
    # each individual of the current generation is replaced by the fittest (best) of three individuals
    # drawn randomly from the current generation
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox

def main():
    toolbox = toolboxInitializer()

    random.seed(64)

    # CXPB is probability with which two individuals are crossed
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB, NGEN = 0.5, 0.2, 50

    print("Start of evolution")

    # pop will be a list composed of 300 individuals
    pop = toolbox.population(n=300)
    # To evaluate this brand new population
    fitnesses = list(map(toolbox.evaluate, pop))

    print("Evaluated {} individuals".format(len(pop)))

    # We first map() the evaluation function to every individual, then assign their respective fitness
    # fitnesses and population are the same order.
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # The Appeal of Evolution
    # Begin the evolution
    for g in range(NGEN):
        print("-- Generation {} --".format(g))

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # we replace the old population by the offspring
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5

        print("Min {}".format(min(fits)))
        print("Max {}".format(max(fits)))
        print("Avg {}".format(mean))
        print("Std {}".format(std))

        if max(fits) == 100:
            break

    print("End of (successful) evolution")
    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is {}, {}".format(best_ind, best_ind.fitness.values))

if __name__ == "__main__":
    main()
