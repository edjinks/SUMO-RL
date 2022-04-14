import pandas as pd
import env
import optparse
import time


# EXPERIMENT SETUPS
# 1) 50 vehicles, all EGOISTS, random routes, til CONVERGENCE
# 2) 50 vehic;es, all PROSCOIALISTS, random routes, til CONVERGENCE
# 3) 50% population EGOISTS, 50% PROSOCIALISTS, random routes
# 4) FCFS Time Taken
# 5) Give Way To Right 


def get_options():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--gui",
                          dest="gui",
                          default=False,
                          help="run the commandline version of sumo")
    opt_parser.add_option("--exp",
                          dest="experimentNumber", 
                          type="int",
                          help="choose experiment")
    opt_parser.add_option("--eps",
                          dest="episodes",
                          type="int",
                          default=5,
                          help="Choose number of episodes")
    options, args = opt_parser.parse_args()
    return options


# 1)
def experiment1(eps, gui):
    print("RUNNING EXP_1...")
    params = {
        'EGOISTS': 100,
        'PROSOCIALISTS': 0,
        'START_EPSILON_DECAY':0.8,
        'END_EPSILON_DECAY':0.1,
        'EPISODES':eps,
        'LEARNING_RATE':0.2,
        'DISCOUNT_FACTOR':0.5,
    }
    return env.run(params, gui)

# 2)
def experiment2(eps, gui):
    print("RUNNING EXP_2...")
    params = {
        'EGOISTS': 0,
        'PROSOCIALISTS': 100,
        'START_EPSILON_DECAY':0.8,
        'END_EPSILON_DECAY':0.1,
        'EPISODES':eps,
        'LEARNING_RATE':0.2,
        'DISCOUNT_FACTOR':0.5,
    }
    return env.run(params, gui)


####### SAVE RESULTS OF LEARNING #######
options = get_options()

if options.experimentNumber == 1:
    policy, rewards, stepsPerEpisode, avgWaits = experiment1(options.episodes, options.gui)
if options.experimentNumber == 2:
    policy, rewards, stepsPerEpisode, avgWaits = experiment2(options.episodes, options.gui)
policyfilename = "results/POLICY_EXP_"+str(options.experimentNumber)+'_'+time.strftime("%Y%m%d-%H%M%S")+".csv"
policy.to_csv(policyfilename)
learningStats = pd.DataFrame({'Rewards':rewards,'Steps':stepsPerEpisode, 'WaitTime':avgWaits})
filename = "results/LEARNING_EXP_"+str(options.experimentNumber)+'_'+time.strftime("%Y%m%d-%H%M%S")+".csv"
learningStats.to_csv(filename)

#########################################



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

