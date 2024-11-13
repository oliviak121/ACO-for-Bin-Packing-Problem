import random
from numpy.random import choice
import numpy as np 
import time
import matplotlib.pyplot as plt 
import pandas as pd
import csv
import os


def initisalise_pheromones_graph(item_list, b):
    """
    Initialises a construction graph with random pheromone values between 0 and 1
    Parameters:
        item_list (list): An array holding the weights of each item
        b (int): The number of bins
    Returns:
        dic: A dictionary representing pheromone levels for transitions between item-bin parings
    """
    pheromones_graph = {}
    pheromones_graph['S'] = {} # Intialise start with with 'S'

    # Creating edges for each item in the item_list except the last
    for item_num, item in enumerate(item_list):
        for bin_num in range(1, b + 1):
            current_key = (bin_num, item)
            pheromones_graph[current_key] = {}
            
            # Starting construction graph from Start node and connecting to first bin
            if item_num == 0:
                pheromones_graph['S'][current_key] = random.uniform(0,1)
            
            # Connecting the last bin to the End node 
            if item_num == len(item_list) -1:
                pheromones_graph[current_key]['E'] = random.uniform(0,1)

            if item_num != len(item_list) -1:
                #connecting current (bin, item) to next (bin, next_item)
                for next_bin_num in range (1, b + 1):
                    next_item = item_list[item_num +1]
                    next_key = (next_bin_num, next_item) 
                    pheromones_graph[current_key][next_key] = random.uniform(0,1)
    
    return pheromones_graph

def choose_next_bin(possible_next_bins, epsilon):
    """
    Chooses the next bin based on a mix or randomness and pheromone strength
    Parameters:
        possible_next_bins (dict): A dictionary of next possible bins with the pheromone strengths on those edges
        epsilon (float): Probability thrshold to choose a bin randomly versus based on pheromone strength
    Returns:
        tuple: The next bin selected in a path, in the form of (bin, item)
    """
    if np.random.rand() < epsilon:
        # Choose a random bin
        next_state = random.choice(list(possible_next_bins.keys()))
    else:
        # Choose based on pheromone strength
        cumulative_sum = 0
        # Calculate the total pheromones for the cumulaative probability
        total_pheromones = sum(possible_next_bins.values())
        random_value = random.uniform(0, total_pheromones)
        
        next_state = None
        for next_bin, pheromone in possible_next_bins.items():
            cumulative_sum += pheromone
            if random_value <= cumulative_sum:
                next_state = next_bin
                break 
    
    return next_state

def construct_ant_path(pheromones_graph, item_list, b, max_iterations, simulate_annealing=False):
    """
    Constructs a path for an ant by selecting a bin for each item based on pheromone levels
    Parameters:
        pheromones_graph (dict): The construction graph with the pheromone values for each item-bin edge
        Item_list (list): An array holding the weights of each item
        b (int): The number of bins
    Returns:
        list: A path the ant traverses, where each element is a tuple (bin, item)
    """
    path = []
    current_state = 'S' #Starts at S
    initial_epsion = 0.2
    final_epsilon = 0
    epsilon_decay = (initial_epsion - final_epsilon) / max_iterations

    # Traversing items in the list
    for item_num in range(len(item_list)):
        if current_state not in pheromones_graph:
            break 

        # The possible next states and thier corresponding pheromones
        possible_next_bins = pheromones_graph[current_state]

        # When reached end node, break
        if 'E' in possible_next_bins:
            break

        next_bin = None
        if simulate_annealing == True:
            # Calculate dynamic epsilon
            epsilon = max(final_epsilon, initial_epsion - item_num * epsilon_decay)
            # Choose next bin 
            next_bin = choose_next_bin(possible_next_bins, epsilon)
            path.append(next_bin)
            current_state = next_bin
        else:
            #Calculate the total pheromones for the cumulaative probability
            total_pheromones = sum(possible_next_bins.values())
            random_value = random.uniform(0, total_pheromones)

            # Choose next bin 
            cumulative_sum = 0
            for next_bin, pheromone in possible_next_bins.items():
                cumulative_sum += pheromone
                if random_value <= cumulative_sum:
                    # print("next" , next_bin)
                    path.append(next_bin)
                    current_state = next_bin
                    break 
        
    return path

def check_convergence(paths):
    """
    Checks if the algorith has converged to a solution by comparing if the last two paths are at least 70% simular
    Parameters:
        paths (list): A list of previously travelled ant paths 
    Returns:
        bool: A True or False value depending on if the algorithm has converged or not
    """
    last = paths[-2]
    second_last = paths[-3]

    # Calculate the total number of tuples in the shorter list
    min_length = min(len(last), len(second_last))

    # Count how many tuples are the same
    match_count = sum(1 for x in last if x in second_last)

    # Calculate the similarity percentage
    similarity_percentage = (match_count / min_length) * 100

    # Check if the similarity is above 70%
    return similarity_percentage >= 70

def construct_ant_path_generation(pheromones_graph, p, item_list, b, allow_extended, simulate_annealing=False):
    """
    Constructs the paths for a populaiton of ants
    Parameters:
        pheromones_graph (dict): The construction graph with the pheromone values for each item-bin edge
        p (int): The number of ants
        item_list (list): An array holding the weights of each item
        b (int): The number of bins
    Returns:
        list of lists: A list where each element is a path generated by an ant
    """
    paths = []
    for _ in range(p):
        path = construct_ant_path(pheromones_graph, item_list, b, p, simulate_annealing)
        
        paths.append(path)
        if allow_extended == True:
            if len(paths) > 2:
                simulate_annealing = check_convergence(paths)

    return paths 


def calculate_fitness(path, b):
    """
    Calculates the fitness of an ant's path, which is the difference between the resulting heaviest and lightest bins total weight
    Parameters:
        path (list): A path the ant transferses, where each element is the bin for the corresponding item
        b (int): The number of bins
    Returns:
        float: The fitness value, which is the difference between the resulting heaviest and lightest bins total weight
    """
    bin_weights = [0] * b # Tnitialse bin weights to zero

    # Distribute the item's weights into the corresponding bins
    for bin_num, item_weight in path:
        if bin_num == 'E': # Skip if reached end node
            continue
        bin_weights[bin_num -1] += item_weight # Update weight of corresponding bin

        # Calculate the fitness as difference between the heaviest and lightest bins 
        fitness = max(bin_weights) - min(bin_weights)

    return fitness

def evaluate_population_fitness(paths, b):
    """
    Evaluates the fitness of each ant's path in generation and finds the best-performing path
    Parameters:
        paths (list of lists): A list of paths generated by ants in the generation
        b (int): The number of bins
    Returns:
        tuple: The best path and its corresponding fitness value
    """
    best_fitness = float('inf') # Initialised with very high value
    best_path = None

    for path in paths:
        # Calculate fitness of path
        fitness = calculate_fitness(path, b)

        # If fitness of path is better than best_path then update
        if fitness < best_fitness:
            best_fitness = fitness
            best_path = path 

    return best_path, best_fitness

def pheromone_update(pheromones_graph, paths, fitnesses, e):
    """
    Updates the pheromone levels in the construction graph based on the ants paths and their fitness values
    Perameters:
        pheromones_graph (dict): Construction graph of pheromones values for each item-bin pair before the update 
        paths (list of lists): A list of paths generated by ants in the generation
        fitness (list of floats): The fitness values corresponding to each path
        e (float): The rate of pheromone evaporation 
    Returns: 
        dict: The updated construction graph with the new pheromone values
    """
    
    # Pheromone reinforcement: increase pheromone values based on path fitness
    for path, fitness in zip(paths, fitnesses):
        pheromone_update = 100/max(fitness, 1e-6) # The better the fitness the higher the pheromone deposit, also prevents division by zero

        # Update pheromones along the path
        for i in range(len(path) -1):
            current_key = path[i]
            next_key = path[i +1]
            pheromones_graph[current_key][next_key] += pheromone_update

    # Pheromone evaporation: decreases all pheromone values my multiplying by evaporation rate
    for key in pheromones_graph:
        for sub_key in pheromones_graph[key]:
            pheromones_graph[key][sub_key] *= e 

    return pheromones_graph
    
def evaluate_generation(ant_paths, b):
    """
    Evaluates the fitness of each ant's path in a generation
    Parameters:
        ant_paths (list of lists): A list where each element is a path traversed by an ant
        b (int): The number of bins
    Returns:
        list: A list of fitness values for each ant's path
    """
    fitnesses = [calculate_fitness(path, b) for path in ant_paths]
    return fitnesses

def aco_bin_packing(item_list, b, p, e, max_evaluations, allow_extended):
    """
    The main ACO loop for the Bin Packing Probelem using ACO
    Parameters:
        item_list (list): An array holding the weights of each item
        b (int): The number of bins
        p (int): The number of ants (paths) per generation
        e (float): The evaporation rate
        max_evaluations (int): The number of fitness evaluations to run ACO algorithm before quiting
    Returns:
        dict: Dictionary containing:
            - 'all_generations_data': List of dicts with fitness values for each generation
            - 'best_overall_path': Best path found across all generations
            - 'best_overall_fitness': Best fitness value found across all generations
            - 'worst_overall_fitness': Worst fitness value found across all generations
    """
    # initialse pheromone graph
    pheromones_graph = initisalise_pheromones_graph(item_list, b)
    current_eval = 0 # To track num of fitness evaluations 
    all_generations_data = []

    best_overall_fitness = float('inf')
    worst_overall_fitness = float('-inf')
    best_overall_path = None

    # Main loop for running ACO untill max evaluations is reached
    while current_eval < max_evaluations:

        # Construct each path for population
        ant_paths = construct_ant_path_generation(pheromones_graph, p, item_list, b, allow_extended)
        generation_fitnesses = evaluate_generation(ant_paths, b)
        # Update total fitness evaluations
        current_eval += len(ant_paths)

        # Update generation stats
        gen_best_fitness = min(generation_fitnesses)
        gen_worst_fitness = max(generation_fitnesses)

        # Update best and worst fitnesses and paths
        for path, fitness in zip(ant_paths, generation_fitnesses):
            if fitness <best_overall_fitness:
                best_overall_fitness = fitness
                best_overall_path = path
            if fitness > worst_overall_fitness:
                worst_overall_fitness = fitness

        # Update generational data
        all_generations_data.append({
            'generation': current_eval // p,
            'best_fitness': gen_best_fitness,
            'worst_fitness': gen_worst_fitness,
            'best_fitness_so_far': best_overall_fitness
        })

        if allow_extended:
            half = max_evaluations / 2
            two_third = (max_evaluations*2)/3
            if current_eval != 0: 
                # First half of evals run with the p and e values inputted into the function
                if current_eval >= half and current_eval <= two_third:
                    # The next 1/6 of evals run with these p and e values
                    p = 55
                    e = 0.5
                if current_eval >= two_third:
                    # The final 1/3 of evals run with these p and e values
                    p = 10
                    e =0.1

        # Update pheromones graph
        pheromones_graph = pheromone_update(pheromones_graph, ant_paths, generation_fitnesses, e)

    return{
        'all_generations_data': all_generations_data,
        'best_overall_path': best_overall_path,
        'best_overall_fitness': best_overall_fitness,
        'worst_overall_fitness': worst_overall_fitness
    }


def run_multiple_trials(item_list, b, p, e, max_evaluations, num_trials, config_label, allow_extended):
    """
    Runs multiple trials (5 as specified in coursework) for an ACO configuration
    Parameters:
        item_list (list): An array holding the weights of each item
        b (int): The number of bins
        p (int): The number of ants
        e (float): The evaporation rate
        max_evaluations (int): The maxiumum number of fitness evaluations 
        num_trials (int): The number of trials to run
        config_label (str): Label describing the ACO configurations
    Returns:
        list: A list of results for each trial
    """
    trial_results = []
    start_time = time.time()
    
    for trial in range(num_trials):
        trial_seed = trial+1 # So trials across ACO types can have the same seed
        random.seed(trial_seed)

        print(f"Starting trial {trial +1} with seed {trial_seed}")
        trial_start_time = time.time()

        trial_data = aco_bin_packing(item_list, b, p, e, max_evaluations, allow_extended)

        trial_end_time = time.time()
        trial_duration = trial_end_time - trial_start_time
        print(f"trial {trial+1} completed in {trial_duration:.2f} seconds")

        trial_results.append({
            'trial': trial + 1,
            'best_fitness': trial_data['best_overall_fitness'],
            'worst_fitness': trial_data['worst_overall_fitness'],
            'all_generations_data': trial_data['all_generations_data']
        })

        #output results to CSV
        output_trial_results_to_csv(trial+1, trial_data['all_generations_data'], config_label)

    total_duration = time.time() - start_time
    print(f"All {num_trials} completed in {total_duration:.2f} seconds")

    return trial_results 

def run_all_aco_variants(item_list, b, max_evaluations, num_trials, allow_extended):
    """
    Runs all the ACO with multiple trials for each for one BPP type
    Parameters:
        item_list (list): An array holding the weights of each item 
        b (int): The number of bins
        max_evaluations (int): The maxiumum number of fitness evaluations 
        num_trials (int): The number of trials to run
    Returns:
        dict: A dictionary with results for all the ACO configurations
    """
    aco_configs = [
        (100, 0.90, "p=100, e=0.90"),  
        (100, 0.60, "p=100, e=0.60"),  
        (10, 0.90, "p=10, e=0.90"),   
        (10, 0.60, "p=10, e=0.60")    
    ]

    all_results = {}
    BPP_start_time = time.time()

    for p, e, config_lable in aco_configs:
        print(f"\nRunning ACO with {config_lable}")
        config_start_time = time.time()

        config_results = run_multiple_trials(item_list, b, p, e, max_evaluations, num_trials, config_lable, allow_extended)
        all_results[config_lable] = config_results

        config_end_time = time.time()
        config_duration = config_end_time - config_start_time
        print(f"ACO configuration {config_lable} completed in {config_duration}")

    BPP_end_time = time.time()
    overall_duration = BPP_end_time - BPP_start_time
    print(f"BPP completed in {overall_duration:.2f} seconds")

    # Ouput all results to CSV
    output_all_results_to_csv(all_results)

    return all_results

def output_trial_results_to_csv(trial, all_generations_data, config_label):
    """
    Outputs generational fitness data to a CSV file for each trial
    Parameters:
        trial (int): The trial number
        all_generations_data (list of dicts): Each dictionary contains detailed fitness information for each generation
        config_label (str): Label describing the ACO configurations
    Returns:
        None
    """
    # Format the config label so it works when plugging it into an output file name
    config_label_formatted = config_label.replace(", ", " ").replace("=", "-").replace(".", "_")
    filename = f"{config_label_formatted}_trial_{trial}.csv"

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Generation', 'Best Fitness', 'Worst Fitness', 'Best Fitness So Far'])
        for data in all_generations_data:
            writer.writerow([
                data['generation'], data['best_fitness'], data['worst_fitness'], data['best_fitness_so_far']])


def output_all_results_to_csv(all_results):
    """
    Outputs all ACO resuls to a CSV file
    Prameters:
        all_results (dict): Dictionary containing all results from ACO configurations
    Returns:
        None
    """
    with open('all_aco_results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Configuration', 'Trial', 'Best Fitness', 'Worst Fitness'])
        for config_label, results in all_results.items():
            for result in results:
                writer.writerow([config_label, result['trial'], result['best_fitness'], result['worst_fitness']])


def plot_fitness_scatter(base_filename, num_trials):
    """
    Plots the performance of a single ACO configuration by plotting fitness over generations using a scatter plot
    Parameters:
        base_filename (str): The base name of the file without the trial number and axtension
        num_trials (int): The number of trials to include in the plot
    """
    fig, ax = plt.subplots(figsize=(8,6))
    all_data = []
    
    for trial in range(1, num_trials +1):
        filename = f'{base_filename}_trial_{trial}.csv'
        if os.path.exists(filename):
            data = pd.read_csv(filename)
            ax.scatter(data['Generation'], data['Best Fitness'], label=f'Trial {trial}', alpha=0.6, s=10)
            all_data.append(data['Best Fitness'])
        else:
            print(f"Warning: File '{filename}' not found and will be skipped")

    # Calculate the average best fitness per generation if data is available
    if all_data:
        all_data = np.array(all_data)
        mean_fitness = np.mean(all_data, axis=0)
        # Fit and plot a regression line to the average data
        coeffs = np.polyfit(data['Generation'], mean_fitness, deg=1)
        regression_line = np.polyval(coeffs, data['Generation'])
        ax.plot(data['Generation'], regression_line, 'r--', label='Line of\nBest Fit', color='black')

    config_label_reformatted = base_filename.replace("-", "=").replace("_", ".")
    ax.set_title(f'Performance for {config_label_reformatted}', fontweight='bold')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    ax.grid(True)
    ax.set_xlim((0, len(data['Generation'])))
    ax.legend(loc='upper right', frameon=True, fontsize='small')

    plt.tight_layout()
    plt.savefig(f"scatter fitness {config_label_reformatted}.png")
    plt.show()

def plot_best_fitness():
    """
    Plots the best fitnesses across configuratins with bars for each of the 5 trials
    Parameters:
        None
    Returns:
        None
    """
    config_labels = ["p=100, e=0.90", "p=100, e=0.60", "p=10, e=0.90", "p=10, e=0.60"]
    best_fitnesses = []

    # Plotting setup
    fig, ax = plt.subplots(figsize=(10, 8))
    indices = np.arange(len(config_labels))
    bar_width = 0.17
    x = np.arange(len(config_labels))
    bar_array = np.zeros((5, len(config_labels)))
    multiplier = 0

    # Collecting best fitness data
    for i in range(len(config_labels)):
        config_fitnesses = []
        config_label_formatted = config_labels[i].replace(", ", " ").replace("=", "-").replace(".", "_")
        
        for trial in range(1, 6):
            filename = f'{config_label_formatted}_trial_{trial}.csv'
            if os.path.exists(filename):
                data = pd.read_csv(filename)
                best_fitness = (data['Best Fitness'].min())
                bar_array[trial-1][i] = best_fitness

    # Plotting each trial     
    for i in range(5):
        rects = ax.bar(x + bar_width*multiplier -bar_width*2, bar_array[i], bar_width, label = f'Trial {i+1}')
        ax.bar_label(rects, padding=3)
        multiplier +=1

    ax.set_xlabel('Configuration')
    ax.set_ylabel('Best Fitness')
    ax.set_title('Best fitness Across Configurations and Trials')
    ax.legend(loc="lower right")

    plt.xticks(x, config_labels)
    plt.tight_layout()
    plt.savefig("Best_Fitness_Comparison.png")
    plt.show()

#TESTING

bpp1_items = [i for i in range(1, 501)] # Item list for BPP1
bpp2_items = [(i**2)/2 for i in range (1, 501)] # Item list for BPP2
bpp1_b = 10 # Num of bins for BPP1
bpp2_b = 50 # Num of bins for BPP2
max_evaluations = 10000 # The number of fitness evaluations
num_trials = 5 # Num of trials


def running(experiemnt, graph):
    """
    Runs the algorithm depending on experiement type choice 
    Parameters:
        experiement (int): 1 for BPP1, 2 for BPP2, 3 for further experiments
        graph (bool): True results in the function producing graphs for the experiement 
    Returns:
        None
    """
    configs = [
                'p-100 e-0_90',
                'p-100 e-0_60',
                'p-10 e-0_90',
                'p-10 e-0_60'
            ]
    if experiemnt == 1:
        run_all_aco_variants(bpp1_items, bpp1_b, max_evaluations, num_trials, allow_extended=False)
        if graph == True:
            for name in configs:
                plot_fitness_scatter(name, 5)
            plot_best_fitness()

    elif experiemnt == 2:
        run_all_aco_variants(bpp2_items, bpp2_b, max_evaluations, num_trials, allow_extended=False)
        if graph == True:
            for name in configs:
                plot_fitness_scatter(name, 5)
            plot_best_fitness()


    elif experiemnt == 3: 
        run_multiple_trials(bpp1_items, bpp1_b, p=10, e=0.9, max_evaluations=20000, num_trials=5, config_label='further research', allow_extended=True)
        if graph == True:
            plot_fitness_scatter('further research', 5)
            return

    else:
        print('That is not a valid input')
        return


# First vairable: input 1 to run BPP1, 2 for BPP2, 3 for further experiements
# Second variable: input True to run graphs 
# The results can be seen in csv files created with the config label 
running(1, True)




