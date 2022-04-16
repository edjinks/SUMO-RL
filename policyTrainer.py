import pandas as pd
import env
import optparse
import time


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
def egoTraining(eps, gui):
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
def proTraining(eps, gui):
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
policyfilename = "policy/POLICY_EXP_"+str(options.experimentNumber)+'_'+time.strftime("%Y%m%d-%H%M%S")+".csv"
policy.to_csv(policyfilename)
learningStats = pd.DataFrame({'Rewards':rewards,'Steps':stepsPerEpisode, 'WaitTime':avgWaits})
filename = "learningData/LEARNING_EXP_"+str(options.experimentNumber)+'_'+time.strftime("%Y%m%d-%H%M%S")+".csv"
learningStats.to_csv(filename)

