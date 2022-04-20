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

def run(ego_q_table, pro_q_table, policyName, egoVehicleNumber, proVehicleNumber, gui=False, routeOrder=None):
    sumoBinary = sumolib.checkBinary('sumo')
    if gui:
        sumoBinary = sumolib.checkBinary('sumo-gui')
    traci.start([sumoBinary, "-c", "network/grid.sumocfg", "--no-warnings", "--tripinfo-output", policyName])
    routeNames = route_creator()
    addVehicles(routeNames, egoVehicleNumber, proVehicleNumber, routeOrder)
    cols = 0
    traci.simulationStep()
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        states = getStates(getLeadersAtJunctions(getLaneLeaders()))
        actions = getActions(states, ego_q_table, pro_q_table)
        doActions(actions)
        cols += traci.simulation.getCollidingVehiclesNumber()
    traci.close() 
    print(cols)


def getWaitTimes(file):
    waitTimes = []
    for veh in sumolib.xml.parse(file, 'tripinfo'):
        waitTimes.append(float(veh.waitingTime))
    return waitTimes

def mixedPolicy(ego_policy_csv, pro_policy_csv, egoNum, proNum, routeOrder=None, gui=False, outName = 'policyTrials/mixedOUT.csv'):
    ego_q_table = pd.read_csv(ego_policy_csv)
    pro_q_table = pd.read_csv(pro_policy_csv)
    run(ego_q_table, pro_q_table, 'temp/Out.xml', egoNum, proNum, gui, routeOrder)
    waitTimes = getWaitTimes('temp/Out.xml')
    df = pd.DataFrame({'wait':waitTimes})
    df.to_csv(outName)
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
    quantiles = [0, 0, 95,99]
    analysisString = "{}: Skew: {}, Kurtois Skew: {}, Quantles: {}%: {}, {}%: {}, {}%: {}, {}%: {}"
    for i in range(len(policies)):
        q = np.percentile(policies[i], quantiles)
        print(analysisString.format(titles[i], stats.skew(policies[i]), stats.kurtosis(policies[i]), quantiles[0], q[0], quantiles[1], q[1], quantiles[2], q[2], quantiles[3], q[3]))
    plotter.compareHistograms(20, policies, titles, quantiles)

#GUI INSPECT POLICY
# run(pd.read_csv('policies/egoPolicy20.csv'), None, 'temp/Out.xml', 500, 0, True, None)

vehNumber = 5000

### CHOOSE A NEW ROUTE ######
# routeOrder = []
# routeNames = route_creator(False)
# for _ in range(vehNumber):
#     routeOrder.append(np.random.choice(routeNames))
# savedRoute = pd.DataFrame({'routeOrder': routeOrder})
# savedRoute.to_csv('policyTrials/routeOrder.csv')

routeOrderDF = pd.read_csv('policyTrials/routeOrder.csv')
routeOrder = list(routeOrderDF['routeOrder'])

egoPolicy = 'policies/OLDegoPolicy.csv'
proPolicy =  'policies/proPolicy20.csv'

# # #### UPDATE RESULTS ######
#egoPolicyDF = singlePolicy(egoPolicy, vehNumber, 'policyTrials/egoOUT.csv', routeOrder)
#proPolicyDF = singlePolicy(proPolicy, vehNumber, 'policyTrials/proOUT.csv', routeOrder)
# mixedPolicyDF = mixedPolicy(egoPolicy, proPolicy, int(vehNumber*0.8), int(vehNumber*0.2), routeOrder)
# fcfsPolicyDF = firstComefirstServed(vehNumber, 0, False, routeOrder)
# gwrDF = giveWayToRight(vehNumber, False, routeOrder)
# fcfsSwapPolicyDF = firstComefirstServed(int(vehNumber*0.8), int(vehNumber*0.2), False, routeOrder, True)

egoPolicyDF = pd.read_csv('policyTrials/egoOUT.csv')['wait']
proPolicyDF = pd.read_csv('policyTrials/proOUT.csv')['wait']
# mixedPolicyDF = pd.read_csv('policyTrials/mixedOUT.csv')['wait']
fcfsPolicyDF = pd.read_csv('policyTrials/fcfsOUT.csv')['wait']
gwrPolicyDF = pd.read_csv('policyTrials/gwrOUT.csv')['wait']
# fcfsSwapPolicyDF = pd.read_csv('policyTrials/fcfsSwapOUT.csv')['wait']

# # comparePolicies([egoPolicyDF, proPolicyDF], ['ego', 'pro'])
# comparePolicies([egoPolicyDF, proPolicyDF, mixedPolicyDF, fcfsPolicyDF, gwrPolicyDF, fcfsSwapPolicyDF], ['ego', 'pro', 'mixed', 'fcfs', 'gwr', 'fcfsSwap'])


# ######### EXPERIMENT 1: EGO vs GWR, FCFS
# comparePolicies([egoPolicyDF, gwrPolicyDF, fcfsPolicyDF], ['ego', 'gwr', 'fcfs'])

# # ######### EXPERIMENT 2: PRO vs GWR, FCFS
# comparePolicies([proPolicyDF, gwrPolicyDF, fcfsPolicyDF], ['pro', 'gwr', 'fcfs'])

# # ######### EXPERIMENT 3: EGO vs PRO
# comparePolicies([egoPolicyDF, proPolicyDF], ['ego', 'pro'])
# comparePolicies([egoPolicyDF, proPolicyDF, gwrPolicyDF, fcfsPolicyDF], ['ego', 'pro', 'gwr', 'fcfs'])

# ######### EXPERIMENT 4: MIXED at VARIOUS PERCENTAGES majority EGO

# m50mixedPolicyDF = mixedPolicy(egoPolicy, proPolicy, int(vehNumber*0.5), int(vehNumber*0.5), routeOrder, False, '50Mixed.csv')
# m60mixedPolicyDF = mixedPolicy(egoPolicy, proPolicy, int(vehNumber*0.6), int(vehNumber*0.4), routeOrder, False, '60Mixed.csv')
# m70mixedPolicyDF = mixedPolicy(egoPolicy, proPolicy, int(vehNumber*0.7), int(vehNumber*0.3), routeOrder, False, '70Mixed.csv')
# m80mixedPolicyDF = mixedPolicy(egoPolicy, proPolicy, int(vehNumber*0.8), int(vehNumber*0.2), routeOrder, False, '80Mixed.csv')
# m90mixedPolicyDF = mixedPolicy(egoPolicy, proPolicy, int(vehNumber*0.9), int(vehNumber*0.1), routeOrder, False,'90Mixed.csv')

m50mixedPolicyDF = pd.read_csv('50Mixed.csv')['wait']
m60mixedPolicyDF = pd.read_csv('60Mixed.csv')['wait']
m70mixedPolicyDF = pd.read_csv('70Mixed.csv')['wait']
m80mixedPolicyDF = pd.read_csv('80Mixed.csv')['wait']
m90mixedPolicyDF = pd.read_csv('90Mixed.csv')['wait']
e_collisions = [998, 1004, 1138, 1154, 1122]
#comparePolicies([m50mixedPolicyDF, m60mixedPolicyDF, m70mixedPolicyDF, m80mixedPolicyDF, m90mixedPolicyDF], ['50','60','70','80','90'])

# ######### EXPERIMENT 5: MIXED at VARIOUS PERCENTAGES majority PRO

# p50mixedPolicyDF = mixedPolicy(egoPolicy, proPolicy, int(vehNumber*0.5), int(vehNumber*0.5), routeOrder, False, 'p50Mixed.csv')
# p60mixedPolicyDF = mixedPolicy(egoPolicy, proPolicy, int(vehNumber*0.4), int(vehNumber*0.6), routeOrder, False, 'p60Mixed.csv')
# p70mixedPolicyDF = mixedPolicy(egoPolicy, proPolicy, int(vehNumber*0.3), int(vehNumber*0.7), routeOrder, False, 'p70Mixed.csv')
# p80mixedPolicyDF = mixedPolicy(egoPolicy, proPolicy, int(vehNumber*0.2), int(vehNumber*0.8), routeOrder, False, 'p80Mixed.csv')
# p90mixedPolicyDF = mixedPolicy(egoPolicy, proPolicy, int(vehNumber*0.1), int(vehNumber*0.9), routeOrder, False, 'p90Mixed.csv')

p50mixedPolicyDF = pd.read_csv('p50Mixed.csv')['wait']
p60mixedPolicyDF = pd.read_csv('p60Mixed.csv')['wait']
p70mixedPolicyDF = pd.read_csv('p70Mixed.csv')['wait']
p80mixedPolicyDF = pd.read_csv('p80Mixed.csv')['wait']
p90mixedPolicyDF = pd.read_csv('p90Mixed.csv')['wait']
p_collisions = [990, 1030, 930, 912, 1122]
#comparePolicies([p50mixedPolicyDF, p60mixedPolicyDF, p70mixedPolicyDF, p80mixedPolicyDF, p90mixedPolicyDF], ['50','60','70','80','90'])


comparePolicies([egoPolicyDF, m90mixedPolicyDF, m80mixedPolicyDF, m70mixedPolicyDF, m60mixedPolicyDF, p50mixedPolicyDF, p60mixedPolicyDF, p70mixedPolicyDF, p80mixedPolicyDF, p90mixedPolicyDF, proPolicyDF], ['100', '90', '80', '70', '60', '50','40','30','20','10','0'])


# ######### EXPERIMENT 5: MIXED OPTIMAL? vs FCFS, GWR
# optimalMixedPolicyDF = mixedPolicy(egoPolicy, proPolicy, int(vehNumber*0.8), int(vehNumber*0.2), routeOrder, '80Mixed.csv')
# comparePolicies([optimalMixedPolicyDF, gwrPolicyDF, fcfsPolicyDF], ['pro', 'gwr', 'fcfs'])

# ######### EXPERIMENT 6: FCFSswap AT VARIOUS PERCENTAGES
# 50fcfsSwapPolicyDF = firstComefirstServed(int(vehNumber*0.5), int(vehNumber*0.5), False, routeOrder, True)
# 60fcfsSwapPolicyDF = firstComefirstServed(int(vehNumber*0.6), int(vehNumber*0.4), False, routeOrder, True)
# 70fcfsSwapPolicyDF = firstComefirstServed(int(vehNumber*0.7), int(vehNumber*0.3), False, routeOrder, True)
# 80fcfsSwapPolicyDF = firstComefirstServed(int(vehNumber*0.8), int(vehNumber*0.2), False, routeOrder, True)
# 90fcfsSwapPolicyDF = firstComefirstServed(int(vehNumber*0.9), int(vehNumber*0.1), False, routeOrder, True)




# egoPolicy1DF = singlePolicy('policies/ego19.csv', vehNumber, 'policyTrials/ego1OUT.csv', routeOrder)
# egoPolicy2DF = singlePolicy('policies/egoPolicy.csv', vehNumber, 'policyTrials/ego2OUT.csv', routeOrder)
#egoPolicy3DF = singlePolicy('policies/egoPolicy20 copy.csv', vehNumber, 'policyTrials/ego3OUT.csv', routeOrder)
# egoPolicy4DF = singlePolicy('policies/NOCRASHegoPolicy.csv', vehNumber, 'policyTrials/ego4OUT.csv', routeOrder)
# egoPolicy5DF = singlePolicy('policies/OLDegoPolicy.csv', vehNumber, 'policyTrials/ego5OUT.csv', routeOrder)
# egoPolicy6DF = singlePolicy('policies/POLICY_EXP_1_20220417-112215.csv', vehNumber, 'policyTrials/ego6OUT.csv', routeOrder)
# egoPolicy7DF = singlePolicy('policies/POLICY_EXP_1_20220418-091855.csv', vehNumber, 'policyTrials/ego7OUT.csv', routeOrder)

# egoPolicy2DF = pd.read_csv('policyTrials/ego2OUT.csv')['wait']
# egoPolicy1DF = pd.read_csv('policyTrials/ego1OUT.csv')['wait']
# egoPolicy3DF = pd.read_csv('policyTrials/ego3OUT.csv')['wait']
# egoPolicy4DF = pd.read_csv('policyTrials/ego4OUT.csv')['wait']
# egoPolicy5DF = pd.read_csv('policyTrials/ego5OUT.csv')['wait']
# egoPolicy6DF = pd.read_csv('policyTrials/ego6OUT.csv')['wait']
# egoPolicy7DF = pd.read_csv('policyTrials/ego7OUT.csv')['wait']



# comparePolicies([egoPolicy1DF, egoPolicy2DF, egoPolicy3DF, egoPolicy5DF, egoPolicy6DF, egoPolicy7DF], ['19','ego','20#','OLD','POL_18','POL_17'])



# comparePolicies([egoPolicy6DF, egoPolicy5DF], ['18','old'])
# comparePolicies([egoPolicy3DF, egoPolicy5DF], ['20','OLD'])
# comparePolicies([egoPolicy6DF, egoPolicy7DF], ['POL_18','POL_17'])



# proPolicy1DF = singlePolicy('policies/OLDproPolicy.csv', vehNumber, 'policyTrials/ego1OUT.csv', routeOrder)
# proPolicy2DF = singlePolicy('policies/pro19.csv', vehNumber, 'policyTrials/ego2OUT.csv', routeOrder)
# proPolicy3DF = singlePolicy('policies/proPolicy20.csv', vehNumber, 'policyTrials/ego3OUT.csv', routeOrder)
# proPolicy4DF = singlePolicy('policies/proPolicy.csv', vehNumber, 'policyTrials/ego4OUT.csv', routeOrder)


# proPolicy1DF = pd.read_csv('policyTrials/ego1OUT.csv')['wait']
# proPolicy2DF = pd.read_csv('policyTrials/ego2OUT.csv')['wait']
# proPolicy3DF = pd.read_csv('policyTrials/ego3OUT.csv')['wait']
# proPolicy4DF = pd.read_csv('policyTrials/ego4OUT.csv')['wait']




# comparePolicies([proPolicy2DF, proPolicy3DF], ['19', '20'])
