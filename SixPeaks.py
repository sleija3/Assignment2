import datetime

import mlrose_hiive as mlrose
import numpy as np
import pandas as pd
import CustomSixPeaks as csp
import six
import sys
sys.modules['sklearn.externals.six'] = six



# kwargs = {'t_pct': .3}
fitness = csp.CustomSixPeaks()
problem = mlrose.DiscreteOpt(length=40, fitness_fn=fitness , maximize=True, max_val=2)
# fitness.get_counter()
# print(fitness.get_counter())
init_state = np.array(np.random.randint(low=0, high=2, size=40))

print("initial state: ", init_state)
iters = 1000 #40 for both
iters_gen = 1000 #67
iters_mimic = 200#300 score of 50 len 30 -- 28 at 200
times = []
t1 = datetime.datetime.now()
# best_state_hill, best_fitness_hill, curve_hill= mlrose.random_hill_climb(problem, max_attempts=iters,
#                                     restarts=10, init_state=init_state, curve=True, max_iters=iters, random_state=1)
# schedule = mlrose.ArithDecay(decay=.000888888)
# best_state_anneal, best_fitness_anneal, fitness_curve_anneal = mlrose.simulated_annealing(problem, schedule = schedule,
#                                                       max_attempts = iters, max_iters = iters,
#                                                       init_state = init_state, curve=True, random_state=1)

# best_state_genetic, \
# best_fitness_genetic, \
# fitness_curve_genetic = mlrose.genetic_alg(problem, max_attempts = iters_gen,max_iters = iters_gen,
#                                            curve=True,pop_size=1000, random_state=1#,
#                                            # state_fitness_callback=my_callback,
#                                            # callback_user_info=[times, 'stephen', iters]
#                                            )
best_state_mimic, \
best_fitness_mimic, \
fitness_curve_mimic = mlrose.mimic(problem, max_attempts = iters_mimic,max_iters = iters_mimic,curve=True,pop_size=1000,
                                           random_state=1, keep_pct = .2#,
                                            # state_fitness_callback=my_callback,
                                            # callback_user_info=[times, 'stephen', iters]
                                   )

# times = pd.DataFrame(index=iters_mimic, columns=iters_mimic,
#                      data=np.zeros(shape=(iters_mimic.shape[0], iters_mimic.shape[0])))
# all_iterations = [1, 2]
# for iteration in all_iterations:
# times = []
# best_state_mimic, \
# best_fitness_mimic, \
# fitness_curve_mimic = mlrose.mimic(problem, max_attempts = 5,max_iters = 5,curve=True,pop_size=600,
#                                            random_state=1, keep_pct = .2, state_fitness_callback=my_callback,
#                                             callback_user_info=[times, 'stephen', 5])
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
# print("curve mimic: ", fitness_curve_mimic)
print("Evals Hill: ", fitness.get_counter())
# print(times)

# print("temp at end of annealing: ", schedule.evaluate(len(fitness_curve_anneal)))
# print("number of iteerations for anealling: ", len(fitness_curve_anneal))

# best_state_hill, best_fitness_hill, curve_hill= mlrose.random_hill_climb(problem, max_attempts=20,
#                                     restarts=20, init_state=init_state, curve=True, max_iters=160, random_state=1)
# schedule = mlrose.ArithDecay(decay=.00999)
# best_state_anneal, best_fitness_anneal, fitness_curve_anneal = mlrose.simulated_annealing(problem, schedule = schedule,
#                                                       max_attempts = 20, max_iters = 160,
#                                                       init_state = init_state, curve=True, random_state=1)
# best_state_genetic, \
# best_fitness_genetic, \
# fitness_curve_genetic = mlrose.genetic_alg(problem, max_attempts = 20,max_iters = 80,curve=True,pop_size=600,
#                                            random_state=1)
# best_state_mimic, \
# best_fitness_genetic_mimic, \
# fitness_curve_mimic = mlrose.mimic(problem, max_attempts = 20,max_iters = 80,curve=True,pop_size=600,
#                                            random_state=1, keep_pct = .8)