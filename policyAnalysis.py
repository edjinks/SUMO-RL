import sumolib
import pandas as pd
import traci
from env import computeReward, route_creator, addVehicles, getLeadersAtJunctions, getLaneLeaders, getStates
import plotter
import numpy as np
from scipy import stats

def doActions(actions):
    vehToGo = -1
    if sum([int(a) for a in actions.values()]) == 0: #all stay stationary
        vehToGo = np.random.randint(0,4)
    count = 0
    for veh in actions.keys():
        if count == vehToGo:
            actions[veh] = '1'
        if actions[veh] == '1':
            traci.vehicle.resume(veh)
            traci.vehicle.setSpeedMode(veh, 32)
        count += 1

def getActions(states, ego_q_table, pro_q_table):
    actions = {}
    for agent in states.keys():
        if 'EGO' in agent:
            q_table = ego_q_table
        elif 'PRO' in agent:
            q_table = pro_q_table
        col = q_table[states[agent]]
        if col.idxmax() == col.idxmin():
            action = np.random.choice(["1", "0"])
        else:
            action = col.idxmax()
        actions.update({agent: str(action)})
    return actions

def giveWayToRight(vehNumber, gui=True, routeOrder=None):
    sumoBinary = sumolib.checkBinary('sumo')
    if gui:
        sumoBinary = sumolib.checkBinary('sumo-gui')
    traci.start([sumoBinary, "-c", "network/grid.sumocfg", "--tripinfo-output", 'temp/Out.xml', "--no-warnings",])
    routeNames = route_creator()
    addVehicles(routeNames, vehNumber, 0, routeOrder)
    traci.simulationStep()
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        states = getStates(getLeadersAtJunctions(getLaneLeaders()))
        if len(states) == 4:
            agent = (list(states.keys())[np.random.randint(0,4)])
            traci.vehicle.resume(agent)
            traci.vehicle.setSpeedMode(agent, 32)
        else:
            for (agent, state) in states.items():
                if state[3] == '0':
                    traci.vehicle.resume(agent)
                    traci.vehicle.setSpeedMode(agent, 32)
                    break
    traci.close()
    policyWait = getWaitTimes('temp/Out.xml')
    df = pd.DataFrame({'wait':policyWait})
    df.to_csv('policyTrials/gwrOUT.csv')
    return df

def swapper(queue, currentlyAtjn):
    #IMPLEMENT SWAP - IF TWO CONSECUTIVE AGENTS IN QUEUE WOULD RECIEVE AN INCREASE IN REWARDS THEN ACCEPT SWAP
    actionDict = {str(i):'0' for i in currentlyAtjn}
    actionDict.update({queue[0]:'1'})
    currentRewA = computeReward(queue[0], actionDict)
    currentRewB = computeReward(queue[1], actionDict)
    actionDict.update({queue[0]:'0', queue[1]: '1'})
    swappedRewA = computeReward(queue[0], actionDict)
    swappedRewB = computeReward(queue[1], actionDict)
    if swappedRewA>=currentRewA and swappedRewB>currentRewB or swappedRewA>currentRewA and swappedRewB>=currentRewB:
        queue[0], queue[1] = queue[1], queue[0]
    return queue

def firstComefirstServed(egoNumber, proNumber=0, gui=True, routeOrder=None, swap=False):
    sumoBinary = sumolib.checkBinary('sumo')
    if gui:
        sumoBinary = sumolib.checkBinary('sumo-gui')
    traci.start([sumoBinary, "-c", "network/grid.sumocfg", "--no-warnings", "--tripinfo-output", 'temp/Out.xml'])
    routeNames = route_creator()
    addVehicles(routeNames, egoNumber, proNumber, routeOrder)
    queue = []
    traci.simulationStep()
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        currentlyAtjn  = getLeadersAtJunctions(getLaneLeaders()).values()
        for agent in currentlyAtjn:
            if agent not in queue:
                queue.append(agent)
        if len(queue)>1 and swap:
            queue = swapper(queue, currentlyAtjn)
        if len(queue)>0:
            traci.vehicle.resume(queue[0])
            traci.vehicle.setSpeedMode(queue[0], 32)
            queue.pop(0)
    traci.close()
    policyWait = getWaitTimes('temp/Out.xml')
    df = pd.DataFrame({'wait':policyWait})
    if swap:
        df.to_csv('policyTrials/fcfsSwapOUT.csv')
    else:
        df.to_csv('policyTrials/fcfsOUT.csv')
    return df

def run(ego_q_table, pro_q_table, policyName, egoVehicleNumber, proVehicleNumber, gui=True, routeOrder=None):
    sumoBinary = sumolib.checkBinary('sumo')
    if gui:
        sumoBinary = sumolib.checkBinary('sumo-gui')
    traci.start([sumoBinary, "-c", "network/grid.sumocfg", "--no-warnings", "--tripinfo-output", policyName])
    routeNames = route_creator()
    addVehicles(routeNames, egoVehicleNumber, proVehicleNumber, routeOrder)
    traci.simulationStep()
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        states = getStates(getLeadersAtJunctions(getLaneLeaders()))
        actions = getActions(states, ego_q_table, pro_q_table)
        doActions(actions)
    traci.close() 

def getWaitTimes(file):
    waitTimes = []
    for veh in sumolib.xml.parse(file, 'tripinfo'):
        waitTimes.append(float(veh.waitingTime))
    return waitTimes

def mixedPolicy(ego_policy_csv, pro_policy_csv, egoNum, proNum, routeOrder=None, gui=False):
    ego_q_table = pd.read_csv(ego_policy_csv)
    pro_q_table = pd.read_csv(pro_policy_csv)
    run(ego_q_table, pro_q_table, 'temp/Out.xml', egoNum, proNum, gui, routeOrder)
    waitTimes = getWaitTimes('temp/Out.xml')
    df = pd.DataFrame({'wait':waitTimes})
    df.to_csv('policyTrials/mixedOUT.csv')
    return df

def singlePolicy(policyCSV, vehicles, waitCSVName, routeOrder=None, gui=False):
    policy = pd.read_csv(policyCSV)
    run(policy, None, 'temp/Out.xml', vehicles, 0, gui, routeOrder)
    policyWait = getWaitTimes('temp/Out.xml')
    df = pd.DataFrame({'wait':policyWait})
    df.to_csv(waitCSVName)
    return df

def policyInspect(policyCSV):
    policy = pd.read_csv(policyCSV)
    run(policy, None, 'temp/Out.xml', 100, 0, True)

def comparePolicies(policies, titles):
    quantiles = [5,10,90,95]
    analysisString = "{}: Skew: {}, Kurtois Skew: {}, Quantles: {}%: {}, {}%: {}, {}%: {}, {}%: {}"
    for i in range(len(policies)):
        q = np.percentile(policies[i], quantiles)
        print(analysisString.format(titles[i], stats.skew(policies[i]), stats.kurtosis(policies[i]), quantiles[0], q[0], quantiles[1], q[1], quantiles[2], q[2], quantiles[3], q[3]))
    plotter.compareHistograms(40, policies, titles, quantiles)


vehNumber = 5000

#### CHOOSE A NEW ROUTE ######
# routeOrder = []
# routeNames = route_creator(False)
# for _ in range(vehNumber):
#     routeOrder.append(np.random.choice(routeNames))
# savedRoute = pd.DataFrame({'routeOrder': routeOrder})
# savedRoute.to_csv('policyTrials/routeOrder.csv')

routeOrderDF = pd.read_csv('policyTrials/routeOrder.csv')
routeOrder = list(routeOrderDF['routeOrder'])


##### UPDATE RESULTS ######
egoPolicyDF = singlePolicy('policies/POLICY_EXP_1_20220418-091855.csv', vehNumber, 'policyTrials/egoOUT.csv', routeOrder)
proPolicyDF = singlePolicy('policies/proPolicy.csv', vehNumber, 'policyTrials/proOUT.csv', routeOrder)
mixedPolicyDF = mixedPolicy('policies/POLICY_EXP_1_20220418-091855.csv', 'policies/proPolicy.csv', int(vehNumber*0.8), int(vehNumber*0.2), routeOrder)
fcfsPolicyDF = firstComefirstServed(vehNumber, 0, False, routeOrder)
gwrDF = giveWayToRight(vehNumber, False, routeOrder)
fcfsSwapPolicyDF = firstComefirstServed(int(vehNumber*0.5), int(vehNumber*0.5), False, routeOrder, True)

egoPolicyDF = pd.read_csv('policyTrials/egoOUT.csv')['wait']
proPolicyDF = pd.read_csv('policyTrials/proOUT.csv')['wait']
mixedPolicyDF = pd.read_csv('policyTrials/mixedOUT.csv')['wait']
fcfsPolicyDF = pd.read_csv('policyTrials/fcfsOUT.csv')['wait']
gwrPolicyDF = pd.read_csv('policyTrials/gwrOUT.csv')['wait']
fcfsSwapPolicyDF = pd.read_csv('policyTrials/fcfsSwapOUT.csv')['wait']


comparePolicies([egoPolicyDF, proPolicyDF, mixedPolicyDF, fcfsPolicyDF, gwrPolicyDF, fcfsSwapPolicyDF], ['ego', 'pro', 'mixed', 'fcfs', 'gwr', 'fcfsSwap'])
