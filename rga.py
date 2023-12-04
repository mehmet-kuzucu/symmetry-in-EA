import numpy as np
import matplotlib.pyplot as plt

def rga():
    np.random.seed()  

    # uper and lower bounds
    op = {'ub': 50.0, 'lb': -50.0}

    gas = {'n_individuals': 500, 'generations': 100, 'p_c': 1.0, 'p_m': 0.9,
           'eta_c': 10, 'eta_m': 1.0, 'variance_generations': 10,
           'verbose': True, 'fIdx': {'fit': 1, 'ref': 2}}

    # INITIALIZATION
    variance_array = np.zeros(gas['n_individuals'])
    queue = np.zeros(gas['variance_generations'])
    

    qIndex = 0
    variance = 0

    # TODO: Check if this code should be here or before the initialization of the variables
    # Ensure even number of individuals
    if gas['n_individuals'] % 2 != 0:
        gas['n_individuals'] += 1

    # Ensure at least one individual
    if gas['n_individuals'] <= 0:
        gas['n_individuals'] = 1

    # RANDOM INITIALIZATION
    pop = initialize_random_population(op, gas)
    
    # EVALUATION
    fit_array_P = evaluate(pop, gas)

    # plot_pop(pop, 'bo', True)       !!

    # ITERATIONS
    for gen in range(1, gas['generations'] + 1):
        # SELECTION
        matPool = selection(fit_array_P, 2, True, gas)
    
        # plot_pop(pop, 'bo', True)       !!
        # plot_mat_pool(pop, matPool)     !!

        # VARIATION
        offspring = variation(pop, matPool, op, gas)
        # plot_pop(offspring, 'ko', False)  !!

        # EVALUATION
        fit_array_O = evaluate(offspring, gas)

        # SURVIVOR
        pop, fit_array_P = survivor(pop, offspring, fit_array_P, fit_array_O, gas)

        # Calculate variance over the last 'varianceGen' generations
        queue[qIndex] = fit_array_P[0, 0]
        qIndex = (qIndex + 1) % gas['variance_generations']


        
        variance = np.var(np.trim_zeros(queue))

        #TODO: check this below code 
        '''trimmed_array = np.trim_zeros(queue)
        if len(trimmed_array) > 1:
            variance = np.var(trimmed_array)
        else:
            variance = 0  # or handle the case where variance is not meaningful
        variance_array[gen-1] = variance'''

        # VERBOSE (SHOW LOG)
        if gas['verbose']:
            print(f"{gen})\tFit: {fit_array_P[0, gas['fIdx']['fit']]:.3f}\n")

        # plot_pop(pop, 'bo', False)     !!

        # Stop if the variance is 0.0000
        if round(variance, 3) == 0 and gen > gas['variance_generations']:
            break

    # Place a breakpoint here to pause and check how the individuals are evolving
    # by plotting the best one with 'drawProblem2D(decodeIndividual(pop[:, :, 0]))'
    return pop, fit_array_P





def initialize_random_population(op:dict, gas:dict) -> np.ndarray:

    pop = np.zeros((gas['n_individuals'], 2))
    for i in range(gas['n_individuals']):
        pop[i, 0] = np.random.uniform(op['lb'], op['ub'])
        pop[i, 1] = np.random.uniform(op['lb'], op['ub'])

    return pop

def rosenbrock(x1, x2):
    return 100.0 * (x2 - x1**2)**2 + (1 - x1)**2



# TODO: ask fabio why the indexis are like this, and why index 0 is empty
def evaluate(pop:np.array, gas:dict) -> np.ndarray:

    #TODO: ask fabio why it is 3 cols
    fit_array = np.zeros((gas['n_individuals'], 3), dtype=np.float32)
    for i in range(gas['n_individuals']):
        
        x1 = pop[i, 0]
        x2 = pop[i, 1]
        fit_array[i, gas['fIdx']['fit']] = rosenbrock(x1, x2)  # objective function
        fit_array[i, gas['fIdx']['ref']] = i

    return fit_array


def selection(fit_array:np.array, k:int, is_min:bool, gas:dict) -> np.ndarray:
    
    mat_pool = np.zeros(gas['n_individuals'], dtype=int)
    for i in range(gas['n_individuals']):
        best_fit = 0
        winner = 0
        for j in range(k):
            index = np.random.randint(0, gas['n_individuals'])
            if j == 0:
                best_fit = fit_array[index, gas['fIdx']['fit']]
                winner = fit_array[index, gas['fIdx']['ref']]
            else:
                if is_min:
                    # for minimization problems
                    if best_fit > fit_array[index, gas['fIdx']['fit']]:
                        best_fit = fit_array[index, gas['fIdx']['fit']]
                        winner = fit_array[index, gas['fIdx']['ref']]
                else:
                    # for maximization problems
                    if best_fit < fit_array[index, gas['fIdx']['fit']]:
                        best_fit = fit_array[index, gas['fIdx']['fit']]
                        winner = fit_array[index, gas['fIdx']['ref']]
        mat_pool[i] = winner

    return mat_pool


def mutation(o_original:np.ndarray, op:dict, gas:dict) -> np.ndarray:

    o_mutated = o_original.copy()

    if np.random.rand() <= gas['p_m']:
        r = np.random.rand()
        delta = 0.0

        if r < 0.5:
            delta = (2.0 * r)**(1.0 / (gas['eta_m'] + 1)) - 1.0
        else:
            delta = 1.0 - (2.0 * (1 - r))**(1.0 / (gas['eta_m'] + 1))

        o_mutated = o_original + (op['ub'] - op['lb']) * delta

        o_mutated[0] = np.clip(o_mutated[0], op['lb'], op['ub'])
        o_mutated[1] = np.clip(o_mutated[1], op['lb'], op['ub'])

    return o_mutated

def variation(pop:np.ndarray, mat_pool:np.ndarray, op:dict, gas:dict) -> np.ndarray:

    # Declare a static array of chromosomes filled with zeros
    offspring = np.zeros((gas['n_individuals'], 2))

    mat_pool = np.random.permutation(mat_pool)  # Shuffle the mating pool

    # This cannot be parallelized (or can it?)
    for i in range(0, gas['n_individuals'], 2):

        # Crossover
        index_p1 = mat_pool[i]
        index_p2 = mat_pool[i + 1]

        p1 = pop[index_p1, :]
        p2 = pop[index_p2, :]

        o1, o2 = crossover(p1, p2, op, gas)

        # Mutation
        o1 = mutation(o1, op, gas)
        o2 = mutation(o2, op, gas)

        offspring[i, :] = o1
        offspring[i + 1, :] = o2

    return offspring

def survivor(pop, offspring, fit_array_P, fit_array_O, gas:dict):
    
    new_pop = np.vstack((pop, offspring))
    new_fit_array = np.vstack((fit_array_P, fit_array_O))

    for i in range(gas['n_individuals']):
        new_fit_array[i, gas['fIdx']['ref']] += gas['n_individuals']

    new_fit_array = new_fit_array[new_fit_array[:, gas['fIdx']['fit']].argsort()]

    new_fit_array = new_fit_array[:gas['n_individuals']]

    for i in range(gas['n_individuals']):
        if new_fit_array[i, gas['fIdx']['ref']] <= gas['n_individuals']:
            pop[i, :] = pop[int(new_fit_array[i, gas['fIdx']['ref']]) - 1, :]
        else:
            pop[i, :] = offspring[int(new_fit_array[i, gas['fIdx']['ref']]) - gas['n_individuals'] - 1, :]

    for i in range(gas['n_individuals']):
        new_fit_array[i, gas['fIdx']['ref']] = i + 1

    fit_array_P = new_fit_array

    return pop, fit_array_P


def crossover(p1:np.ndarray, p2:np.ndarray, op:dict, gas:dict) -> np.ndarray:

    o1 = p1.copy()
    o2 = p2.copy()

    if np.random.rand() <= gas['p_c']:
        k = np.random.rand()
        beta = 0.0

        if k <= 0.5:
            beta = (2.0 * k)**(1.0 / (gas['eta_c'] + 1))
        else:
            beta = (1 / (2.0 * (1.0 - k)))**(1.0 / (gas['eta_c'] + 1))

        o1 = 0.5 * ((p1 + p2) - beta * (p2 - p1))
        o2 = 0.5 * ((p1 + p2) + beta * (p2 - p1))

        o1[0] = np.clip(o1[0], op['lb'], op['ub'])
        o1[1] = np.clip(o1[1], op['lb'], op['ub'])
        o2[0] = np.clip(o2[0], op['lb'], op['ub'])
        o2[1] = np.clip(o2[1], op['lb'], op['ub'])

    return o1, o2

def plot_mat_pool(population, mat_pool, gas:dict):

    plt.plot(population[:, 0], population[:, 1], 'bo')

    for i in range(gas['n_individuals']):
        plt.plot(population[mat_pool[i], 0], population[mat_pool[i], 1], 'go')

    plt.plot(1, 1, 'r+')
    plt.show()

def plot_pop(population, style, is_new, op:dict):
    
    if is_new:
        plt.figure()

    plt.plot(population[:, 0], population[:, 1], style)
    plt.plot(1, 1, 'r+')
    plt.xlim([op['lb'], op['ub']])
    plt.ylim([op['lb'], op['ub']])
    plt.show()


# Example usage:

if __name__ == '__main__':
    pop_result, fit_result = rga()
    print(f"Best individual: {pop_result[0]}, fitness: {fit_result[0, 0]}")
