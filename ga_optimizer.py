from random import random, choice
from math import sin, cos


def ga_random(dimension):
    return tuple([10.0 * random() for _ in range(dimension)])


def create_population(size=100, dimension=2):
    return [ga_random(dimension) for _ in range(size)]


def crossing_genome(population, scale=2):
    new_population = list()
    for father in population:
        for _ in range(scale):
            mother = choice(population)
            # add small mutation
            child = tuple([(f + m) / 2 + random() - 0.5
                for f, m in zip(father, mother)])
            new_population.append(child)
    return new_population


def selection(population, fitness, scale=2):
    size = len(population) // scale
    return sorted(population, key=lambda x: -fitness(*x))[:size]


def next_generation(population):
    return population


def main():
    func = lambda x, y: sin(x) + cos(y)     # noqa

    generations = 10

    population = create_population(100)

    for n in range(generations):
        population = crossing_genome(population)

        population = selection(population, func)

        for i in range(3):
            p = population[i]
            print(p, func(*p))


if __name__ == '__main__':
    main()
