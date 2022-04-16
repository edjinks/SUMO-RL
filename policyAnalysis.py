import sumolib
import pandas as pd
import traci
from env import route_creator, addVehicles, getLeadersAtJunctions, getLaneLeaders, getStates, doActions
import plotter
import numpy as np
from scipy import stats

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
    traci.start([sumoBinary, "-c", "network/grid.sumocfg", "--no-warnings", "--collision.action", 'none', "--tripinfo-output", 'temp/Out.xml'])
    routeNames = route_creator()
    addVehicles(routeNames, vehNumber, 0, routeOrder)
    traci.simulationStep()
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        states = getStates(getLeadersAtJunctions(getLaneLeaders()))
        if len(states) == 4:
            agent = (list(states.keys())[np.random.randint(0,4)])
            traci.vehicle.resume(agent)
        else:
            for (agent, state) in states.items():
                if state[3] == '0':
                    traci.vehicle.resume(agent)
                    break
    traci.close()
    policyWait = getWaitTimes('temp/Out.xml')
    df = pd.DataFrame({'wait':policyWait})
    df.to_csv('policyTrials/gwrOUT.csv')
    return df


def firstComefirstServed(vehNumber, gui=True, routeOrder=None):
    sumoBinary = sumolib.checkBinary('sumo')
    if gui:
        sumoBinary = sumolib.checkBinary('sumo-gui')
    traci.start([sumoBinary, "-c", "network/grid.sumocfg", "--no-warnings","--collision.action", "none", "--tripinfo-output", 'temp/Out.xml'])
    routeNames = route_creator()
    addVehicles(routeNames, vehNumber, 0, routeOrder)
    queue = []
    traci.simulationStep()
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        currentlyAtjn  = getLeadersAtJunctions(getLaneLeaders()).values()
        for agent in currentlyAtjn:
            if agent not in queue:
                queue.append(agent)
        if len(queue)>0:
            traci.vehicle.resume(queue[0])
            queue.pop(0)
    traci.close()
    policyWait = getWaitTimes('temp/Out.xml')
    df = pd.DataFrame({'wait':policyWait})
    df.to_csv('policyTrials/fcfsOUT.csv')
    return df


def run(ego_q_table, pro_q_table, policyName, egoVehicleNumber, proVehicleNumber, gui=True, routeOrder=None):
    sumoBinary = sumolib.checkBinary('sumo')
    if gui:
        sumoBinary = sumolib.checkBinary('sumo-gui')
    traci.start([sumoBinary, "-c", "network/grid.sumocfg", "--no-warnings","--collision.action", "none", "--tripinfo-output", policyName])
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


def mixedPolicyToWaitTimesCSV(ego_policy_csv, pro_policy_csv, egoNum, proNum, routeOrder=None, gui=False):
    ego_q_table = pd.read_csv(ego_policy_csv)
    pro_q_table = pd.read_csv(pro_policy_csv)
    run(ego_q_table, pro_q_table, 'temp/Out.xml', egoNum, proNum, gui, routeOrder)
    waitTimes = getWaitTimes('temp/Out.xml')
    df = pd.DataFrame({'wait':waitTimes})
    df.to_csv('policyTrials/mixedOUT.csv')
    return df


def policyToWaitTimesCSV(policyCSV, vehicles, waitCSVName, routeOrder=None, gui=False):
    policy = pd.read_csv(policyCSV)
    run(policy, None, 'temp/Out.xml', vehicles, 0, gui, routeOrder)
    policyWait = getWaitTimes('temp/Out.xml')
    df = pd.DataFrame({'wait':policyWait})
    df.to_csv(waitCSVName)
    return df

##### EGO, PRO, MIXED, FCFS, GIVEWAYRIGHT COMPARISON #####
vehNumber = 5
routeOrder = []
routeNames = route_creator(False)
for _ in range(vehNumber):
    routeOrder.append(np.random.choice(routeNames))

egoPolicyDF = policyToWaitTimesCSV('policies/egoPolicy.csv', vehNumber, 'policyTrials/egoOUT.csv', routeOrder)
proPolicyDF = policyToWaitTimesCSV('policies/proPolicy.csv', vehNumber, 'policyTrials/proOUT.csv', routeOrder)
mixedPolicyDF = mixedPolicyToWaitTimesCSV('policies/egoPolicy.csv', 'policies/proPolicy.csv', int(vehNumber/2), int(vehNumber/2), routeOrder)
fcfsPolicyDF = firstComefirstServed(vehNumber, False, routeOrder)
gwrDF = giveWayToRight(vehNumber, False, routeOrder)

egoPolicyDF = pd.read_csv('policyTrials/egoOUT.csv')
proPolicyDF = pd.read_csv('policyTrials/proOUT.csv')
mixedPolicyDF = pd.read_csv('policyTrials/mixedOUT.csv')
fcfsPolicyDF = pd.read_csv('policyTrials/fcfsOUT.csv')
gwrPolicyDF = pd.read_csv('policyTrials/gwrOUT.csv')

plotter.compareHistograms([egoPolicyDF['wait'], proPolicyDF['wait'], mixedPolicyDF['wait'], fcfsPolicyDF['wait'], gwrPolicyDF['wait']], ['ego', 'pro', 'mixed', 'fcfs', 'gwr'])
print('Ego: Skew: ', stats.skew(egoPolicyDF['wait']), 'Kurtois Skew: ', stats.kurtosis(egoPolicyDF['wait']))
print('Pro: Skew: ', stats.skew(proPolicyDF['wait']), 'Kurtois Skew: ', stats.kurtosis(proPolicyDF['wait']))
print('Mixed: Skew: ', stats.skew(mixedPolicyDF['wait']), 'Kurtois Skew: ', stats.kurtosis(mixedPolicyDF['wait']))
print('FCFS: Skew: ', stats.skew(fcfsPolicyDF['wait']), 'Kurtois Skew: ', stats.kurtosis(fcfsPolicyDF['wait']))
print('GWR: Skew: ', stats.skew(gwrPolicyDF['wait']), 'Kurtois Skew: ', stats.kurtosis(gwrPolicyDF['wait']))

