import datetime
import CustomQueens as cq

import mlrose_hiive as mlrose
import numpy as np

fitness = cq.Queens()
problem = mlrose.DiscreteOpt(length=18, fitness_fn=fitness , maximize=False, max_val=18)
iters = 1000
iter_gen = 200
iter_mimic = 50
t1 = datetime.datetime.now()
best_state_hill, best_fitness_hill, curve_hill= mlrose.random_hill_climb(problem, max_attempts=iters,
                                    restarts=10, curve=True, max_iters=iters, random_state=2)

# schedule = mlrose.ArithDecay(decay=.001)
# schedule = mlrose.GeomDecay(decay=.999)

# schedule = mlrose.ExpDecay(exp_const=.00277777)
# best_state_anneal, best_fitness_anneal, fitness_curve_anneal = mlrose.simulated_annealing(problem, schedule = schedule,
#                                                       max_attempts = iters, max_iters = iters, curve=True, random_state=2)

#
# best_state_genetic, \
# best_fitness_genetic, \
# fitness_curve_genetic = mlrose.genetic_alg(problem, max_attempts = iter_gen,max_iters = iter_gen,
#                                            curve=True,pop_size=2000,random_state=2)
#
# best_state_mimic, \
# best_fitness_mimic, \
# fitness_curve_mimic = mlrose.mimic(problem, max_attempts = iter_mimic,max_iters = iter_mimic,curve=True,
#                                    pop_size=2000, keep_pct = .2, random_state=2)
t2 = datetime.datetime.now()
t = t2 - t1
print(t)
print("best state hill: ", best_state_hill)
# print("best state anneal: ", best_state_anneal)
# print("best state genetic: ", best_state_genetic)
# print("best state mimic: ", best_state_mimic)
print("best fitness hill: ", best_fitness_hill)
# print("best fitness anneal: ", best_fitness_anneal)
# print("best fitness genetic: ", best_fitness_genetic)
# print("best fitness mimic: ", best_fitness_mimic)
# print("curve hill: ", curve_hill)
# print("curve anneal: ", fitness_curve_anneal)
# print("curve genetic: ", fitness_curve_genetic)
# print("curve mimic: ", fitness_curve_genetic)
print("Evalutations: ", fitness.get_counter())
# print("temp at end of annealing: ", schedule.evaluate(len(fitness_curve_anneal)))
# print("number of iteerations for anealling: ", len(fitness_curve_anneal))