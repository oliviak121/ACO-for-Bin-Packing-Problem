"""
PLAN:
1. initialise pheromone table, 
input: k items, b bins, 
output: pheromone table

2. construct ant paths, 
input: pheromone table, p number of ants, 
output: generages a list (p length) representing a solution to bin packing 

3. selecting which bin to go to next 
input: pheromone values for the current item
output: selected bin

4. calculate fitness of a path (heaviest - lightest bin)
input: a path (list)
output: fitness

5. find best performing path in a generation
input: list of all paths in current generation
output: best path, fitness value

6. Update pheroone table based on fitness of each path (lower fitness = larger 
pheromone boost)
input: pheromone table, list of paths, fitness of each path
output: updated pheromone table

7. pheromone evaporation (happens accross all paths at same rate)
input: pheromone table, evap rate
output: pheromone table after evaporation 

8. main aco loop set up: for each iteration construct ant paths, evaluate 
fitness, update pheromones, evaporates pheromones. terminates after max iterations
or once maz number of fitness evaluation is reached
input: max number of interations, number of ants, evaporation rate
output: best path found, its fitness 

9. run experiments for BPP1 and BPP2
input: none
output: best fitenss for each parameter combination

10. analyse results
input: results from multiple runs
output: insights into which parameters performed the best
"""