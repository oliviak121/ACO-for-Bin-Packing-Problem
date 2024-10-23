import random
from numpy.random import choice
import numpy as np 

"""
p = num of ants, index - n
e = pheromone evaporation rate (0.9 or 0.6)
b = num of bins (10 or 50), index - j
k = num of items (500), index - i (spec has it as n)
w_b = list of weight in bins
w_i = list of item weight
Pij = corresponding probablilites to each entry in T
T = pheromone matrix
SE = set of ant paths from S to E (item in list rangesf rom 1 to 10 representing the bins)

d = fitness of solution - difference in heaviest bin and lighest bin
"""

def GeneratreSE(T, k, b, p):
    SE = [[] for n in range(p)]
    denom = np.sum(T, axis = 1)
    Pij = [T[i]/denom[i] for i in range(k)]
    for n in range(p):
        for i in range(k):
            SE[n].append(choice([j for j in range(b)], p = Pij[i])+1)
    return SE 

def Fitness(SE, k, b, p, w_i):
    d = [0 for n in range(p)] #dif d between heaviest and lighest bins
    for n in range(p):
        w_b = [0 for j in range(b)]
        for i in range(k):
            w_b[SE[n][i]-1] = w_b[SE[n][i]-1] + w_i[SE[n][i]-1]
        d[n] = max(w_b) - min(w_b)
    return d

def GenerateResutls(d, SE):
    results = 

