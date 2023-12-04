import numpy as np
import matplotlib.pyplot as plt

def sea():
    np.random.seed()
    
    # uper and lower bounds
    op = {'ub': 50.0, 'lb': -50.0}

    gas = {'n_individuals': 16, 'generations': 100, 'p_m': 0.9, 'chaos_exp': 30.0,
           'variance_generations': 10, 'verbose': True, 'fIdx': {'fit': 0, 'ref': 1}}

    best = [0, 0, 0]

    # --INITIALIZATION
    variance_array = np.zeros(gas['n_individuals'])
    queue = np.zeros(gas['variance_generations'])  # queue used to calculate the variance of the last 'variance_generations' generations best individuals
    qIndex = 0
    variance = 0

    # in case a funny user decides to have an odd number of individuals in the population...
    if gas['n_individuals'] % 2 != 0:
        gas['n_individuals'] += 1

    # a funnier user...
    if gas['n_individuals'] <= 0:
        gas['n_individuals'] = 1

    # --RANDOM INITIALIZATION
    pop = initialize_random_population(op, gas)

    # --EVALUATION
    fit_array_P = evaluate(pop, gas)

    plot_pop(pop, 'bo', True, op)

    # --ITERATIONS
    for gen in range(1, gas['generations'] + 1):

        if gen % 2 == 1:
            # do symmetry
            offspring = generate_offspring_with_symmetry(pop, op, gas)

            # --EVALUATION
            fit_array_O = evaluate(offspring, gas)

            # --SURVIVOR
            pop, fit_array_P = survivor_elitism(pop, offspring, fit_array_P, fit_array_O, op, gas)
        else:
            # do mutation
            offspring = generate_offspring_with_chaos(pop, op, gas)

            # --EVALUATION
            fit_array_O = evaluate(offspring, gas)

            # --SURVIVOR
            pop, fit_array_P = survivor_non_elitism(offspring, fit_array_O, op, gas)

        # calculate variance over the last 'varianceGen' generations
        queue[qIndex] = fit_array_P[0, 0]  # variance is on ik fitness only
        qIndex += 1
        if qIndex >= len(queue):
            qIndex = 0
        variance = np.var(np.trim_zeros(queue))  # calculate variance
        variance_array[gen-1] = variance

        if gen == 1 or fit_array_P[0, gas['fIdx']['fit']] < best[2]:
            best[:2] = pop[int(fit_array_P[0, gas['fIdx']['ref']]), :]
            best[2] = fit_array_P[0, gas['fIdx']['fit']]

        # --VERBOSE (SHOW LOG)
        if gas['verbose']:
            print(f'{gen})\t{best}')

        plot_pop(pop, 'bo', False, op)

        # stop if the variance is 0.0000
        if round(variance, 3) == 0 and gen > gas['variance_generations']:
            break

    # place a breakpoint here as you run the algorithm to pause
    # and check how the individuals are evolving by plotting the best one with 'drawProblem2D(decodeIndividual(pop(:,:,1)))'
    return best

def plot_pop(pop, style, is_new, op:dict):
    if is_new:
        plt.figure()

    plt.plot(pop[:, 0], pop[:, 1], style)
    plt.plot(1, 1, 'r+')
    plt.xlim([op['lb'], op['ub']])
    plt.ylim([op['lb'], op['ub']])
    plt.show()

def plot_mat_pool(pop, mat_pool, gas:dict):
    plt.figure()
    for i in range(gas['n_individuals']):
        plt.plot(pop[mat_pool[i], 0], pop[mat_pool[i], 1], 'go')
    plt.plot(1, 1, 'r+')
    plt.show()

def initialize_random_population(op:dict, gas:dict) -> np.ndarray:

    pop = np.zeros((gas['n_individuals'], 2))
    for i in range(gas['n_individuals']):
        pop[i, 0] = (op['ub'] - op['lb']) * np.random.rand() + op['lb']
        pop[i, 1] = (op['ub'] - op['lb']) * np.random.rand() + op['lb']
    return pop

def rosenbrock(x1, x2):
    return 100.0 * (x2 - x1**2)**2 + (1 - x1)**2

def evaluate(pop:np.array, gas:dict) -> np.ndarray:

    s = pop.shape[0]
    fit_array = np.zeros((s, 3))
    for i in range(s):
        x1, x2 = pop[i, 0], pop[i, 1]
        fit_array[i, gas['fIdx']['fit']] = rosenbrock(x1, x2)  # objective function
        fit_array[i, gas['fIdx']['ref']] = i + 1

    return fit_array

def generate_offspring_with_symmetry(pop:np.ndarray, op:dict, gas:dict) -> np.ndarray:

    off_pop = np.zeros((0, 2))
    for i in range(gas['n_individuals']):
        off = symmetry(pop[i, :])
        print(f"pop[i, :]: {pop[i, :]}")
        off_pop = np.vstack((off_pop, off))

    for i in range(off_pop.shape[0]):
        off_pop[i, 0] = max(min(off_pop[i, 0], op['ub']), op['lb'])
        off_pop[i, 1] = max(min(off_pop[i, 1], op['ub']), op['lb'])
    return off_pop

def symmetry(ind):
    dv = ind.shape[0]
    if dv == 1:
        combs = combinations([-1, 1])
    elif dv == 2:
        combs = combinations([-1, 1], [-1, 1])
    elif dv == 3:
        combs = combinations([-1, 1], [-1, 1], [-1, 1])

    combs = np.delete(combs, -1, axis=0)
    off = ind * combs
    return off

def generate_offspring_with_chaos(pop:np.ndarray, op:dict, gas:dict) -> np.ndarray:

    off_pop = np.zeros((0, 2))
    for i in range(gas['n_individuals']):
        off = chaos(pop[i, :], op, gas)
        off_pop = np.vstack((off_pop, off))

    return off_pop

def chaos(o_original, op:dict, gas:dict) -> np.ndarray:

    dv = o_original.shape[0]
    v = np.random.randn(dv)
    v = v / np.sqrt(np.dot(v, v))

    r = ((op['ub'] - op['lb']) / gas['chaos_exp']) * np.random.rand()

    o_mutated = o_original + r * v

    o_mutated[0] = max(min(o_mutated[0], op['ub']), op['lb'])
    o_mutated[1] = max(min(o_mutated[1], op['ub']), op['lb'])

    return o_mutated

def survivor_non_elitism(offspring, fit_array_O, op:dict, gas:dict) -> np.ndarray:

    fit_array_O = fit_array_O[fit_array_O[:, gas['fIdx']['fit']].argsort()]

    pop = np.zeros((gas['n_individuals'], 2))
    for i in range(gas['n_individuals']):
        pop[i, :] = offspring[int(fit_array_O[i, gas['fIdx']['ref']]) - 1, :]
    fit_array_P = fit_array_O[:gas['n_individuals'], :]

    return pop, fit_array_P


def survivor_elitism(pop, offspring, fit_array_P, fit_array_O, op:dict, gas:dict):

    new_pop = np.vstack((pop, offspring))
    new_fit_array = np.vstack((fit_array_P, fit_array_O))

    for i in range(gas['n_individuals'], new_pop.shape[0]):
        new_fit_array[i, gas['fIdx']['ref']] += gas['n_individuals']

    sorted_indices = np.argsort(new_fit_array[:, gas['fIdx']['fit']])
    new_fit_array = new_fit_array[sorted_indices, :]

    new_fit_array = new_fit_array[:gas['n_individuals'], :]

    for i in range(gas['n_individuals']):
        if new_fit_array[i, gas['fIdx']['ref']] <= gas['n_individuals']:
            pop[i, :] = pop[int(new_fit_array[i, gas['fIdx']['ref']]) - 1, :]
        else:
            pop[i, :] = offspring[int(new_fit_array[i, gas['fIdx']['ref']]) - gas['n_individuals'], :]

    for i in range(gas['n_individuals']):
        new_fit_array[i, gas['fIdx']['ref']] = i + 1

    fit_array_P = new_fit_array

    return pop, fit_array_P


def combinations(*args):
    if len(args) == 1:
        return np.array([[args[0][0]], [args[0][1]]])
    else:
        combs = combinations(*args[1:])
        new_combs = np.zeros((combs.shape[0] * 2, len(args[0])))
        for i in range(combs.shape[0]):
            new_combs[2 * i, :] = np.hstack((args[0][0], combs[i, :]))
            new_combs[2 * i + 1, :] = np.hstack((args[0][1], combs[i, :]))
        return new_combs
    

if __name__ == '__main__':
    sea()
