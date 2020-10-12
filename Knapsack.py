import datetime

import mlrose_hiive as mlrose
import numpy as np
import CustomKnapsack as ck

# weights = np.array(np.random.randint(low=1, high=30, size=30))
# values = np.array(np.random.randint(low=1, high=30, size=30))
weights = np.array([15, 7, 28, 15, 1, 24, 17, 4, 4, 16, 18, 23, 20, 26, 26, 27, 26, 24, 20, 20, 8, 15, 14, 22, 24, 4
                    ,11, 21, 2, 10])
values = np.array([20, 17, 12, 18, 26, 4, 14, 21, 4, 9, 6, 5, 20, 19, 13, 14, 28, 29, 10, 28, 18, 24, 23, 15,
                   24, 24, 24, 1, 14, 20])
max_weight_pct = 0.3
sum_of_values = values.sum()
sum_of_values_pct = sum_of_values * max_weight_pct
fitness = ck.Knapsack(weights.tolist(), values.tolist(), max_weight_pct)
problem = mlrose.DiscreteOpt(length=30, fitness_fn=fitness , maximize=True, max_val=2)

print("weights:", weights)
print("values: ", values)
print("sum of values: ", sum_of_values)
print("sum of values pct: ", sum_of_values - sum_of_values_pct)
# iters = 1000
iters_gen = 1000
iters_mimic = 600
t1 = datetime.datetime.now()
# best_state_hill, best_fitness_hill, curve_hill= mlrose.random_hill_climb(problem, max_attempts=iters,
#                                     restarts=10, curve=True, max_iters=iters)

# schedule = mlrose.ArithDecay(decay=.001)
# schedule = mlrose.GeomDecay(decay=.999)
# schedule = mlrose.ExpDecay(exp_const=.000555555)
# schedule = mlrose.ExpDecay(exp_const=.00222111)
# best_state_anneal, best_fitness_anneal, fitness_curve_anneal = mlrose.simulated_annealing(problem, schedule = schedule,
#                                                       max_attempts = iters, max_iters = iters, curve=True)

# best_state_genetic, \
# best_fitness_genetic, \
# fitness_curve_genetic = mlrose.genetic_alg(problem, max_attempts = iters_gen,max_iters = iters_gen,
#                                            curve=True,pop_size=1000)

best_state_mimic, \
best_fitness_mimic, \
fitness_curve_mimic = mlrose.mimic(problem, max_attempts = iters_mimic,max_iters = iters_mimic,
                                   curve=True,pop_size=1000, keep_pct = .3)
t2 = datetime.datetime.now()
t = t2 - t1
print(t)
# print("best state hill: ", best_state_hill)
# print("best state anneal: ", best_state_anneal)
# print("best state genetic: ", best_state_genetic)
print("best state mimic: ", best_state_mimic)
# print("best fitness hill: ", best_fitness_hill)
# print("best fitness anneal: ", best_fitness_anneal)
# print("best fitness genetic: ", best_fitness_genetic)
print("best fitness mimic: ", best_fitness_mimic)
# print("curve hill: ", curve_hill)
# print("curve anneal: ", fitness_curve_anneal)
# print("curve genetic: ", fitness_curve_genetic)
# print("curve mimic: ", fitness_curve_genetic)
print("Evaluations: ", fitness.get_counter())

# print("temp at end of annealing: ", schedule.evaluate(len(fitness_curve_anneal)))
# print("number of iteerations for anealling: ", len(fitness_curve_anneal))