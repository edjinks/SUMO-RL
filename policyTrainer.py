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
    opt_parser.add_option("--veh",
                          dest="veh",
                          type="int",
                          default=100,
                          help="Choose number of veh")
    opt_parser.add_option("--nosave",
                          action="store_true",
                          default=False,
                          help="don't save policy and learning details")
    options, args = opt_parser.parse_args()
    return options


# 1)
def egoTraining(eps, gui, veh):
    print("RUNNING EXP_1...")
    params = {
        'EGOISTS': veh,
        'PROSOCIALISTS': 0,
        'START_EPSILON_DECAY':0.8,
        'END_EPSILON_DECAY':0.1,
        'EPISODES':eps,
        'LEARNING_RATE':0.3,
        'DISCOUNT_FACTOR':0.5,
    }
    return env.run(params, gui)

# 2)
def proTraining(eps, gui, veh):
    print("RUNNING EXP_2...")
    params = {
        'EGOISTS': 0,
        'PROSOCIALISTS': veh,
        'START_EPSILON_DECAY':0.8,
        'END_EPSILON_DECAY':0.1,
        'EPISODES':eps,
        'LEARNING_RATE':0.3,
        'DISCOUNT_FACTOR':0.5,
    }
    return env.run(params, gui)


####### SAVE RESULTS OF LEARNING #######
options = get_options()

if options.experimentNumber == 1:
    policy, rewards, stepsPerEpisode, avgWaits = egoTraining(options.episodes, options.gui, options.veh)
if options.experimentNumber == 2:
    policy, rewards, stepsPerEpisode, avgWaits = proTraining(options.episodes, options.gui, options.veh)
if not options.nosave:
    policyfilename = "policies/POLICY_EXP_"+str(options.experimentNumber)+'_'+time.strftime("%Y%m%d-%H%M%S")+".csv"
    policy.to_csv(policyfilename)
    learningStats = pd.DataFrame({'Rewards':rewards,'Steps':stepsPerEpisode, 'WaitTime':avgWaits})
    filename = "learningData/LEARNING_EXP_"+str(options.experimentNumber)+'_'+time.strftime("%Y%m%d-%H%M%S")+".csv"
    learningStats.to_csv(filename)

