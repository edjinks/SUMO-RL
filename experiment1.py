import env
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# EXPERIMENT SETUPS
# 1) 50 vehicles, all EGOISTS, random routes, til CONVERGENCE
# 2) 50 vehic;es, all PROSCOIALISTS, random routes, til CONVERGENCE
# 3) 50% population EGOISTS, 50% PROSOCIALISTS, random routes
# 4) FCFS Time Taken
# 5) Give Way To Right 


# 1)
experiment_params = {
    'EGOISTS': 100,
    'PROSOCIALISTS': 0,
    'START_EPSILON_DECAY':0.8,
    'END_EPSILON_DECAY':0.1,
    'EPISODES':50,
    'LEARNING_RATE':0.2,
    'DISCOUNT_FACTOR':0.5,
}

policy = env.run(experiment_params)
print(policy)
policy.to_csv("results/experiment1.csv")

# 2)

# 3)

# 4)

# def firstComeFirstServed(q):
#     if len(q)>0:
#         veh = q[0]
#         print(veh)
#         traci.vehicle.setSpeedMode(veh, 32)
#         traci.vehicle.setSpeed(veh, 10)
#         q.remove(veh)
#         return q
#     else:
#         return []

# # 5)

# def giveWayToRight(state):
#     if state[3] != '0':
#         vehAction = "0"
#     else:
#         vehAction = "1"
#     return vehAction


############RESULTS PLOTTING######################
def pltResults(steps, waitingTime):
    x = np.array([x for x in range(0,len(waitingTime))])
    y = np.array(steps)
    y2 = np.array(waitingTime)
    print(len(x), len(y), len(y2))
    m, b = np.polyfit(x, y, 1)
    m2, b2 = np.polyfit(x, y2, 1)
    plt.figure(figsize=(20, 5))
    plt.plot(x, y, 'o')
    plt.plot(x, y2, 'o')
    plt.plot(x, m*x+b)
    plt.plot(x, m2*x+b2)
    plt.tight_layout()
    plt.show()