import random
from numpy.random import choice
import numpy as np 
import time
import matplotlib.pyplot as plt 
import contextlib
import cProfile
import pprint


def initisalise_pheromones_graph(k, b):
    """
    Initialises a construction graph with random pheromone values between 0 and 1
    parameters:
        k (int): the number of items
        b (int): the number of bins
    returns:
        dic: a dictionary representing the construction graph of where 
        each key is and the value is a dictionary of bins and their 
        corresponding pheromone values
    """
    pheromones_graph = {}

    for item in range(1, k +1): #loop over each item
        pheromones_graph[item] = {}
        for bin_num in range(1, b + 1): #loop over each bin for the item
            #initilse each edge with a random pheromone value
            pheromones_graph[item][bin_num] = random.uniform(0, 1)

    pprint.pprint(pheromones_graph)
    return pheromones_graph
        

def construct_ant_path(pheromones_graph, k, b):
    """
    Constructs a path for an ant by selecting a bin for each item based on pheromone levels
    parameters:
        pheromones_graph (dict): the construction graph with the pheromone values for each item-bin edge
        k (int): the number of items
        b (int): the number of bins
    returns:
        list: a path the ant transferses, where each element is the bin for the corresponding item
    """
    path = []

    for item in range (1, k + 1):
        # get pheromone values for current item
        pheromones = pheromones_graph[item]

        # calculate the total sum of all the pheromone paths for this item - normalisation
        total_pheromones = sum(pheromones.values())

        # generate random number between 0 and total pheromone for the threshold
        random_threshold = random.uniform(0, total_pheromones)

        # selecting which bin to go to next
        cumulative_pheromone = 0
        selected_bin = 1
        for bin_num in range (1, b + 1):
            cumulative_pheromone += pheromones[bin_num]
            if cumulative_pheromone >= random_threshold:
                selected_bin = bin_num
                break

        # append the selected bin to the path
        path.append(selected_bin)
    
    return path

def construct_ant_path_generation(pheromones_graph, p, k, b):
    """
    Constructs the paths for a populaiton of ants
    Parameters:
        pheromones_graph (dict): the construction graph with the pheromone values for each item-bin edge
        p (int): the number of ants
        k (int): the number of items
        b (int): the number of bins
    Returns:
        list of lists: a list where each element is a path generated by an ant
    """
    paths = []

    for ants in range(p):
        path = construct_ant_path(pheromones_graph, k, b)
        paths.append(path)

    return paths 


def calculate_fitness(path, k, b, weight_function):
    """
    Calculates the fitness of an ant's path, which is the difference between the resulting heaviest and lightest bins total weight
    Parameters:
        path (list): a path the ant transferses, where each element is the bin for the corresponding item
        k (int): the number of items
        b (int): the number of bins
        weight_function (function): function to calculate the weight of an item
    Returns:
        float: the fitness value, which is the difference between the resulting heaviest and lightest bins total weight
    """
    # itisalise bin weights to zero
    bin_weights = [0] * b

    # distribute the item's weights into the corresponding bins
    for item_index, bin_index in enumerate(path):
        # calculate item weight based on weight function 
        item_weight = weight_function(item_index + 1)
        bin_weights[bin_index - 1] += item_weight #update the weight of the corresponding bin 

    # return fitness
    return max(bin_weights) - min(bin_weights)

def evaluate_population_fitness(paths, k, b):
    """
    Evaluates the fitness of each ant's path in generation and finds the best-performing path
    Parameters:
        paths (list of lists): a list of paths generated by ants in the generation
        k (int): the number of items
        b (int): the number of bins
    Returns:
        tuple: the best path and its corresponding fitness value
    """
    best_fitness = float('inf') #initialised with very high value
    best_path = None

    for path in paths:
        #calculate fitness of path
        fitness = calculate_fitness(path, k, b)

        #if fitness of path is better than best_path then update
        if fitness < best_fitness:
            best_fitness = fitness
            best_path = path 

    return best_path, best_fitness

def pheromone_update(pheromones_graph, paths, fitnesses, b, e):
    """
    Updates the pheromone levels in the construction graph based on the ants paths and their fitness values
    Perameters:
        pheromones_graph (dict): construction graph of pheromones values for each item-bin pair before the update 
        paths (list of lists): a list of paths generated by ants in the generation
        fitness (list of floats): the fitness values corresponding to each path
        b (int): the number of bins 
        e (float): the rate of pheromone evaporation 
    Returns: 
        dict: the updated construction graph with the new pheromone values
    """
    
    #pheromone reinforcement: increase pheromone values based on path fitness
    for path, fitness in zip(paths, fitnesses):
        pheromone_update = 100/max(fitness, 1e-6) #the better the fitness the higher the pheromone deposit, also prevents division by zero

        #update pheromones along the path
        for item_index, bin_index in enumerate(path):
            pheromones_graph[item_index + 1][bin_index] += pheromone_update

    #pheromone evaporation: decreases all pheromone values my multiplying by evaporation rate
    for item in pheromones_graph:
        for bin_num in range(1, b + 1):
            pheromones_graph[item][bin_num] *= e 

    return pheromones_graph

def aco_bin_packing(k, b, p, e, max_evaluations, weight_function):
    """
    The main ACO loop for the Bin Packing Probelem using ACO
    Parameters:
        k (int): the number of items
        b (int): the number of bins
        p (int): the number of ants (paths) per generation
        e (float): the evaporation rate
        max_evaluations (int): the number of fitness evaluations to run ACO algorithm before quiting
    Returns:
        tuple: the best, worst, and average fitness values over trial, with total fitness evaluations
    """
    # initialse pheromone graph
    pheromones_graph = initisalise_pheromones_graph(k,b)

    best_overall_path = None
    best_overall_fitness = float('inf')
    worst_overall_fitness = float('-inf')
    total_evaluations = 0 # to track num of fitness evaluations 
    all_generations_data = []

    # main loop for running ACO untill max evaluations is reached
    while total_evaluations < max_evaluations:

        #construct each path for population
        ant_paths = construct_ant_path_generation(pheromones_graph, p, k, b)
        
        #evaluate fitness of each path
        fitnesses = [calculate_fitness(path, k, b, weight_function) for path in ant_paths]
    
        #update total fitness evaluations
        total_evaluations += p

        #find stats for generation
        best_fitness = min(fitnesses)
        worst_fitness = max(fitnesses)
        average_fitness = sum(fitnesses)/p

        #find best path in this generation
        best_path_index = fitnesses.index(best_fitness)
        best_path = ant_paths[best_path_index]

        #update the pheromones based on the fitness
        pheromones_graph = pheromone_update(pheromones_graph, ant_paths, fitnesses, b, e)

        #track the overall best solutions for all generations
        if best_fitness < best_overall_fitness:
            best_overall_fitness = best_fitness
            best_overall_path = best_path

        #track worst overall fitness
        if worst_fitness > worst_overall_fitness:
            worst_overall_fitness = worst_fitness

        #store the generation data 
        all_generations_data.append({
            'total_evaluations': total_evaluations,
            'best_fitness': best_fitness,
            'worst_fitness': worst_fitness,
            'average_fitness': average_fitness
        })

        #print generation stats
        #print(f"  \nBest fitness in generation {total_evaluations/100}: {best_fitness}")
        #print(f"  Worst fitness in generation {total_evaluations/100}: {worst_fitness}")
        #print(f"  Average fitness in generation {total_evaluations/100}: {average_fitness}")
        #print(f"  Best overall fitness so far: {best_overall_fitness}")

    return best_overall_path, best_overall_fitness, worst_overall_fitness, all_generations_data

def weight_pbb1(i):
    """
    Generates weights of item i for BPP1 - item i has weight i
    Parameters:
        i (int): the item index
    Returns:
        list: the weight of the item
    """
    return i

def weight_pbb2(i):
    """
    Generates weights of item i for BPP2 - item i has weight (i^2)/2
    Parameters:
        i (int): the item index
    Returns:
        list: the weight of the item
    """
    return (i ** 2)/2 


def run_single_trial(k, b, p, e, max_evaluations, seed, weight_function):
    """
    Runs a single trial for a specific ACO configuration with fixed seed
    Parameters:
        k (int): the number of items
        b (int): the number of bins
        p (int): the number of ants
        e (float): the evaporation rate
        max_evaluations (int): the maximum number of fitness evaluations
        seed (int): the random seed to use for reproducibility
        weight_function (function): the function to calculate the item weights
    Returns:
        None 
    """
    # set random seed
    random.seed(seed)

    print(f"\nRunning single trial with p={p}, e={e}, seed={seed}")

    #record start time
    start_time = time.time()

    # run the aco algorithm
    best_path, best_fitness, worst_fitness, generations_data = aco_bin_packing(k, b, p, e, max_evaluations, weight_function)

    # calculate average fitness over all generations
    total_fitness_sum = sum(gen_data['average_fitness'] for gen_data in generations_data)
    num_generations = len(generations_data)
    avg_fitness = total_fitness_sum/ num_generations if num_generations > 0 else 0

    #record end time and calculate time elapsed
    end_time = time.time()
    elapsed_time = end_time - start_time

    # output results
    print(f"Best fitness: {best_fitness}")
    print(f"Worst fitness: {worst_fitness}")
    print(f"Average fitness over the trial: {avg_fitness:.2f}")
    
    #print("Generation data:")
    #for gen_data in generations_data:
    #    print(f"Total Evaluations: {gen_data['total_evaluations']}, Best = {gen_data['best_fitness']}, Worst = {gen_data['worst_fitness']}, Average = {gen_data['average_fitness']}")
    
    # output the total time taken to run the trial
    print (f"Time taken for this trial: {elapsed_time:.2f} seconds")

    '''
    #testing plotting best fitness across one trial
    aco_label = (f"p = {p}, e = {e}")
    if weight_function == weight_pbb1:
        bpp_label = "BPP1"
    elif weight_function == weight_pbb2:
        bpp_label = "BPP2"
    else:
        bpp_label = 'ERROR'
    plot_generation_fitness(generations_data, aco_label, bpp_label)
    '''

def run_multiple_trials(k, b, p, e, max_evaluations, num_trials, weight_function):
    """
    Runs multiple trials (usually will be 5) of the same ACO configuration, using a different seed for each trial
    Parameters:
        k (int): the number of items
        b (int): the number of bins
        p (int): the number of ants
        e (float): the evaporation rate
        max_evaluations (int): the maximum number of fitness evaluations per trial
        num_trials (int): number of trials to run
        weight_function (function): the function to calculate the item weights
    Returns:
        None
    """
    #record start time
    start_time = time.time()

    for trial in range(num_trials):
        # generate new random seed for each trial
        trial_seed = random.randint(0, 10000)
        print(f"\nStarting Trial {trial +1} with seed {trial_seed}")
        run_single_trial(k, b, p, e, max_evaluations, trial_seed, weight_function)
    
    #record end time and calculate time elapsed
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken for all {num_trials} trials: {elapsed_time:.2f} seconds")

def run_all_aco_types(k, b, max_evaluations, num_trials, weight_function):
    """
    Runs all ACO configurations (4 types) with 5 trials each for BPP
    Parameters:
        k (int): the number of items 
        b (int): the number of bins 
        max_evaluations (int): The maximum number of fitness evaluations per trial
        num_trials (int): Number of trials to run for each configuration
    Returns:
        None
    """
    # record start time
    start_time = time.time()

    # ACO configurations
    aco_configs = [
        (100, 0.90),  # p = 100, e = 0.90
        (100, 0.60),  # p = 100, e = 0.60
        (10, 0.90),   # p = 10, e = 0.90
        (10, 0.60)    # p = 10, e = 0.60
    ]
    
    #run BPP
    for p, e in aco_configs:
        print(f"\nRunning ACO with p={p}, e={e}: ")
        run_multiple_trials(k, b, p, e, max_evaluations, num_trials, weight_function)

    
    #record end time and calculate time taken to run whole program 
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total time taken for all ACO configurations and trials: {elapsed_time:.2f} seconds")

def plot_generation_fitness(generations_data, aco_label, bpp_label):
    """
    Plots the evolution of the best fitness over generations for one trial
    Parameters:
        generations_data (list): a list of dictionaries, where each dictionary contains data for a generation
    Returns:
        None
    """
    #extracting the fitnesses values from fenerations data
    generations = list(range(1, len(generations_data) + 1))
    best_fitnesses = [gen_data['best_fitness'] for gen_data in generations_data]
    
    #plot over the generations
    plt.figure(figsize=(12, 8))
    plt.plot(generations, best_fitnesses, label='Best Fitness', color="green", marker='o')

    #plot line of best fit
    z = np.polyfit(generations, best_fitnesses, 1)
    p = np.poly1d(z)
    plt.plot(generations, p(generations), label='Line of Best Fit', color='orange', linestyle='--', linewidth= 3)

    plt.xlim([0,len(generations)])
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.title(f"Best Fitness Evolution Over Generations in one trial of {aco_label} in {bpp_label}")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_trials_average_fitness(k, b, max_evalutions, num_trials, weight_function, bpp_label):
    """
    Plots the average best fitnesses over the trials for one BPP type
    Parameters:
        k (int): Number of items
        b (int): Number of bins
        max_evaluations (int): Maximum number of fitness evaluations
        num_trials (int): Number of trials per configuration
        weight_function (function): Function to calculate item weights
        bpp_label (str): Label for BPP type (e.g., "BPP1" or "BPP2")
    Returns:
        None
    """
    # ACO configurations
    aco_configs = [
        (100, 0.90, "p=100, e=0.90"),
        (100, 0.60, "p=100, e=0.60"),
        (10, 0.90, "p=10, e=0.90"),
        (10, 0.60, "p=10, e=0.60")
    ]

    # Setting up subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"Fitness Evolution Over Generations Averaged Over {num_trials} Trials For {bpp_label}", fontsize=16, y=0.98)

    for i, (p, e, config_label) in enumerate(aco_configs):
        best_fitness_per_gen = []

        # Running trials and collecting data
        for trial in range(num_trials):
            best_path, best_fitness, worst_fitness, generations_data = aco_bin_packing(k, b, p, e, max_evaluations, weight_function)

            # Collect fitness data for each generation in this trial
            if trial == 0:
                best_fitness_per_gen = np.array([gen['best_fitness'] for gen in generations_data], dtype=float)
            else:
                # Add data for subsequent trials
                best_fitness_per_gen += np.array([gen['best_fitness'] for gen in generations_data])
        
        # Average over all trials
        best_fitness_per_gen /= num_trials

        # Plotting on the subplot
        ax = axes[i // 2, i % 2]  # Determine subplot position
        generations = list(range(1, len(best_fitness_per_gen) + 1))

        ax.plot(generations, best_fitness_per_gen, label="Best Fitness", color="green", marker='o')
        trend_line = np.poly1d(np.polyfit(generations, best_fitness_per_gen, 1))
        ax.plot(generations, trend_line(generations), color="orange", linestyle="--", linewidth=2, label="Regression")

        ax.set_title(f"ACO Configuration: {config_label}", fontweight = 'bold')
        ax.set_xlabel("Generations")
        ax.set_ylabel("Fitness")
        ax.legend(loc="best")
        ax.grid(True)

    # Adjust layout and show plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig("ACO results")

def plot_single_aco_type(k, b, p, e, max_evaluations, num_trials, weight_function):
    """
    Runs an ACO configuration with 5 trials and plots the best and worst fitnesses
    Parameters:
        k (int): the number of items
        b (int): the number of bins
        p (int): the number of bins
        e (float): the evaportation rate
        max_evaluations (int): the maximum number of fitness evaluations
        num_trials (int): the number of trials to run
        weight_function (function): function to calcualte item weights
    Returns:
        None
    """
    best_fitness_values = []
    worst_fitness_values = []

    for trial in range(num_trials):
        trial_seed = random.randint(0, 1000)
        
        #runs the ACO algorithm for this configuration and trial
        best_path, best_fitness, worst_fitness, generations_data = aco_bin_packing(k, b, p, e, max_evaluations, weight_function)
        
        # Track best, worst, and average fitness for this trial
        best_fitness_values.append(best_fitness)
        worst_fitness_values.append(worst_fitness)
    
    # Plot the results for the 5 trials
    trials = list(range(1, num_trials + 1))
    bar_width = 0.4 

    #setting positons for the bars
    r1 = [x - bar_width for x in trials] #positions for best fit
    r2 = trials #positions for worst fit

    plt.figure(figsize=(10, 6))
    
    #plotting bars
    bars_best = plt.bar(r1, best_fitness_values, label="Best Fitness", color="green", width=bar_width)
    #bars_worst = plt.bar(r2, worst_fitness_values, label="Worst Fitness", color="red", width=bar_width)
    
    #displaying values on top of each bar
    for bar in bars_best:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')

    #for bar in bars_worst:
    #    yval = bar.get_height()
    #    plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')

    plt.xlabel("Trials")
    plt.ylabel("Fitness")
    plt.title(f"Fitness Across Trials for p={p}, e={e}")
    plt.xticks(trials)
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ACO results")
    

def plot_all_aco_type(k, b, max_evaluations, num_trials, weight_function, bpp_label):
    """
    Runs an ACO configuration with 5 trials and plots the best and worst fitnesses
    Parameters:
        k (int): the number of items
        b (int): the number of bins
        max_evaluations (int): the maximum number of fitness evaluations
        num_trials (int): the number of trials to run
        weight_function (function): function to calcualte item weights
        bpp_label (str): label for BPP type
    Returns:
        None
    """
    start_time = time.time()
    print("time started")

    aco_configs = [
        (100, 0.90, "p=100, e=0.90"),
        (100, 0.60, "p=100, e=0.60"),
        (10, 0.90, "p=10, e=0.90"),
        (10, 0.60, "p=10, e=0.60")
    ]

    avg_best_fitness_per_config = {}
    avg_worst_fitness_per_config = {}

    for p, e, config_label in aco_configs:
        best_fitness_values = []
        worst_fitness_values = []

        for trial in range(num_trials):
            trial_seed = random.randint(0, 1000)
            print(f"running trial {trial}")
            
            #runs the ACO algorithm for this configuration and trial
            best_path, best_fitness, worst_fitness, generations_data = aco_bin_packing(k, b, p, e, max_evaluations, weight_function)
            
            # Track best, worst, and average fitness for this trial
            best_fitness_values.append(best_fitness)
            worst_fitness_values.append(worst_fitness)
    
        #calculate average stats across the 5 trials
        avg_best_fitness_per_config[config_label] = sum(best_fitness_values) / num_trials
        avg_worst_fitness_per_config[config_label] = sum(worst_fitness_values) / num_trials

    print("plotting")
    # Plot the results for the 5 trials
    config_label = list(avg_best_fitness_per_config.keys())
    avg_best_fitness = list(avg_best_fitness_per_config.values())
    avg_worst_fitness = list(avg_worst_fitness_per_config.values())

    configs = list(range(len(config_label)))
    bar_width = 0.4

    #setting positons for the bars
    r1 = [x - bar_width/2 for x in configs] #positions for best fit
    r2 = [x + bar_width/2 for x in configs] #positions for worst fit

    plt.figure(figsize=(12, 8))
    
    #plotting bars
    bars_best = plt.bar(r1, avg_best_fitness, label="Average Best Fitness across trials", color="green", width=bar_width)
    #bars_worst = plt.bar(r2, avg_worst_fitness, label="Average Worst Fitness across trials", color="red", width=bar_width)
    
    #display value on top of each bar
    for bar in bars_best:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')
    
    #for bar in bars_worst:
    #    yval = bar.get_height()
    #    plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')

    plt.xticks(configs, config_label, rotation=45,ha='right')
    plt.xlabel("ACO Configurations")
    plt.ylabel("Fitness")
    plt.title(f"Fitness Across Trials ACO Configurations {bpp_label}", weight='bold')
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ACO results")
    # plt.show()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total time taken for all ACO configurations and trials: {elapsed_time:.2f} seconds")
    


#TESTING

k_bpp1 = 500  # Number of items for BPP1
b_bpp1 = 10   # Number of bins for BPP1
k_bpp2 = 500  # Number of items for BPP2
b_bpp2 = 50   # Number of bins for BPP2
max_evaluations = 10000  # Stop after 10,000 fitness evaluations
num_trials = 5  # Run 5 trials

#redirecting output to a file
#with open("test_reults.txt", "w") as f, contextlib.redirect_stdout(f):
    ###testing single trial
    #run_single_trial(k_bpp2, b_bpp2, p=10, e=0.90, max_evaluations=max_evaluations, seed=random.randint(0, 10000), weight_function=weight_pbb2)

    ##testing multiple trials
    #run_multiple_trials(k_bpp1, b_bpp1, p=10, e=0.60, max_evaluations=max_evaluations, num_trials=num_trials, weight_function=weight_pbb1)

    ##testing running all ACO types in one go for one BPP type
    #print("Running ACO for BPP1: ")
    #run_all_aco_types(k_bpp1, b_bpp1, max_evaluations, num_trials, weight_pbb1)
    #print("Running ACO for BPP2: ")
    #run_all_aco_types(k_bpp2, b_bpp2, max_evaluations, num_trials, weight_pbb2)

    ##testing plotting one ACO type
    #plot_single_aco_type(k_bpp1, b_bpp1, p=10, e=0.60, max_evaluations=max_evaluations, num_trials=5, weight_function=weight_pbb1)

    ##testing plotting all ACO types - BPP1
    #plot_all_aco_type(k_bpp1, b_bpp1, max_evaluations, num_trials,weight_pbb1, "BPP1")

    ##testing plotting all ACO types - BPP2
    #plot_all_aco_type(k_bpp2, b_bpp2, max_evaluations, num_trials,weight_pbb2, "BPP2")

    ##testing plotting all average fitnesses for one BPP1
    #plot_trials_average_fitness(k_bpp1, b_bpp1, max_evaluations, num_trials, weight_pbb1, "BPP1")

    ##testing plotting all average fitnesses for one BPP2
    #plot_trials_average_fitness(k_bpp2, b_bpp2, max_evaluations, num_trials, weight_pbb2, "BPP2")

#profiler = cProfile.Profile()
#profiler.enable()
#plot_all_aco_type(k_bpp2, b_bpp2, max_evaluations, num_trials,weight_pbb2, "BPP2")
#profiler.dump_stats("profile_results.prof")
#profiler.disable()

initisalise_pheromones_graph(4,2)