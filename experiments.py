import env
import plotter

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
    'EPISODES':5,
    'LEARNING_RATE':0.2,
    'DISCOUNT_FACTOR':0.5,
}

policy, rewards, stepsPerEpisode = env.run(experiment_params)
policy.to_csv("results/experiment1.csv")
plotter.pltRewardAndSteps(rewards, stepsPerEpisode)

# 2)
experiment_params = {
    'EGOISTS': 0,
    'PROSOCIALISTS': 100,
    'START_EPSILON_DECAY':0.8,
    'END_EPSILON_DECAY':0.1,
    'EPISODES':5,
    'LEARNING_RATE':0.2,
    'DISCOUNT_FACTOR':0.5,
}

policy2, rewards2, stepsPerEpisode2 = env.run(experiment_params)
policy2.to_csv("results/experiment2.csv")

plotter.pltRewardAndSteps(rewards, stepsPerEpisode)

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

