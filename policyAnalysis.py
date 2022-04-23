from turtle import up
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

def updateCSV(csvName, df):
    og = pd.read_csv(csvName)
    merge = pd.concat([og, df], join="inner")
    merge.to_csv(csvName)


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

def giveWayToRight(vehNumber, gui=False, routeOrder=None):
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
    if trials == 0:
        df.to_csv('policyTrials/gwrOUT.csv')
    else:
        updateCSV('policyTrials/gwrOUT.csv', df)
    return df

def getStateValues(veh_id, egoPolicy, proPolicy, state):
    
    if "EGO" in veh_id:
        col = egoPolicy[state]
    if "PRO" in veh_id:
        col = proPolicy[state]
    return col


def swapper(queue, egoPolicy, proPolicy, currentlyAtjn):
    swap = 0
    #IMPLEMENT SWAP - IF TWO CONSECUTIVE AGENTS IN QUEUE WOULD RECIEVE AN INCREASE IN REWARDS THEN ACCEPT SWAP
    actionDict = {str(i):'0' for i in currentlyAtjn.values()}
    actionDict.update({queue[0]:'1'})
    #add value of action in the state of that agent and see if it would be higher if swapper
    states = getStates(currentlyAtjn)
    a = getStateValues(queue[0], egoPolicy, proPolicy, states[queue[0]])
    b = getStateValues(queue[1], egoPolicy, proPolicy, states[queue[1]])
    currentRewA = computeReward(queue[0], actionDict)+a[1]
    currentRewB = computeReward(queue[1], actionDict)+b[0]
    actionDict.update({queue[0]:'0', queue[1]: '1'})
    swappedRewA = computeReward(queue[0], actionDict)+a[0]
    swappedRewB = computeReward(queue[1], actionDict)+b[1]
    if swappedRewA>=currentRewA and swappedRewB>currentRewB or swappedRewA>currentRewA and swappedRewB>=currentRewB:
        queue[0], queue[1] = queue[1], queue[0]
        swap = 1
    return queue, swap

def firstComefirstServed(egoPolicy, proPolicy, egoNumber, proNumber=0, gui=False, routeOrder=None, swap=False, trials=0):
    sumoBinary = sumolib.checkBinary('sumo')
    if gui:
        sumoBinary = sumolib.checkBinary('sumo-gui')
    traci.start([sumoBinary, "-c", "network/grid.sumocfg", "--no-warnings", "--tripinfo-output", 'temp/Out.xml'])
    routeNames = route_creator()
    addVehicles(routeNames, egoNumber, proNumber, routeOrder)
    queue = []
    swaps = 0
    cols = 0
    traci.simulationStep()
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        currentlyAtjnDict  = getLeadersAtJunctions(getLaneLeaders())
        currentlyAtjn = currentlyAtjnDict.values()
        for agent in currentlyAtjn:
            if agent not in queue:
                queue.append(agent)
        if len(queue)>1 and swap:
            queue, swapped = swapper(queue, egoPolicy, proPolicy, currentlyAtjnDict)
            swaps += swapped
        if len(queue)>0:
            traci.vehicle.resume(queue[0])
            traci.vehicle.setSpeedMode(queue[0], 32)
            queue.pop(0)
        cols += traci.simulation.getCollidingVehiclesNumber()
    traci.close()
    policyWait = getWaitTimes('temp/Out.xml')
    df = pd.DataFrame({'wait':policyWait})
    if swap==False:
        if trials == 0:
            df.to_csv('policyTrials/fcfsOUT.csv')
        else:
            updateCSV('policyTrials/fcfsOUT.csv', df)
    return df, swaps, cols

def run(ego_q_table, pro_q_table, policyName, egoVehicleNumber, proVehicleNumber, gui=False, routeOrder=None):
    sumoBinary = sumolib.checkBinary('sumo')
    if gui:
        sumoBinary = sumolib.checkBinary('sumo-gui')
    traci.start([sumoBinary, "-c", "network/grid.sumocfg", "--no-warnings", "--tripinfo-output", policyName])
    routeNames = route_creator()
    addVehicles(routeNames, egoVehicleNumber, proVehicleNumber, routeOrder)
    cols = 0
    traci.simulationStep()
    traci.simulationStep()
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        states = getStates(getLeadersAtJunctions(getLaneLeaders()))
        actions = getActions(states, ego_q_table, pro_q_table)
        doActions(actions)
        cols += traci.simulation.getCollidingVehiclesNumber()
    traci.close() 
    return cols


def getWaitTimes(file):
    waitTimes = []
    for veh in sumolib.xml.parse(file, 'tripinfo'):
        waitTimes.append(float(veh.waitingTime))
    return waitTimes

def mixedPolicy(ego_policy_csv, pro_policy_csv, egoNum, proNum, routeOrder=None, gui=False, outName = 'policyTrials/mixedOUT.csv', trials=0):
    ego_q_table = pd.read_csv(ego_policy_csv)
    pro_q_table = pd.read_csv(pro_policy_csv)
    cols = run(ego_q_table, pro_q_table, 'temp/Out.xml', egoNum, proNum, gui, routeOrder)
    waitTimes = getWaitTimes('temp/Out.xml')
    df = pd.DataFrame({'wait':waitTimes})
    if trials == 0:
        df.to_csv(outName)
    else:
        updateCSV(outName, df)
    return df, cols

def singlePolicy(policyCSV, vehicles, waitCSVName, routeOrder=None, gui=False, trials=0):
    policy = pd.read_csv(policyCSV)
    cols = run(policy, None, 'temp/Out.xml', vehicles, 0, gui, routeOrder)
    policyWait = getWaitTimes('temp/Out.xml')
    df = pd.DataFrame({'wait':policyWait})
    if trials == 0:
        df.to_csv(waitCSVName)
    else:
        updateCSV(waitCSVName, df)
    return df, cols

def policyInspect(policyCSV):
    policy = pd.read_csv(policyCSV)
    run(policy, None, 'temp/Out.xml', 100, 0, True)

def comparePolicies(policies, titles, bins=40):
    quantiles = [90, 95,99]
    analysisString = "{}: Skew: {}, Kurtois Skew: {}, Quantles: {}%: {}, {}%: {}, {}%: {}"
    for i in range(len(policies)):
        q = np.percentile(policies[i], quantiles)
        print(analysisString.format(titles[i], stats.skew(policies[i]), stats.kurtosis(policies[i]), quantiles[0], q[0], quantiles[1], q[1], quantiles[2], q[2]))
    plotter.compareHistograms(bins, policies, titles, quantiles)

def runMixedPercentages(min, max, egoPolicy, proPolicy, vehNum, routeOrder, trials=0):
    collisionsDict = {}
    for i in range (min, max+1):
        egoNum = i*10
        outName = "policyTrials/ego"+str(egoNum)+'Mixed.csv'
        df, collisions = mixedPolicy(egoPolicy, proPolicy, int(vehNum*egoNum/100), int(vehNum*(100-egoNum)/100), routeOrder, False, outName)
        collisionsDict.update({'ego'+str(egoNum): collisions})
    collisionsDF = pd.DataFrame(collisionsDict, index=[0])
    if trials == 0:
        collisionsDF.to_csv('policyTrials/MixedCollisions.csv')
    else:
        updateCSV('policyTrials/MixedCollisions.csv', collisionsDF)


def runMixedPercentagesFCFSSwap(min, max, vehNum, routeOrder, egoPolicy, proPolicy, trials=0):
    swapsDict = {}
    collisionsDict = {}
    for i in range (min, max+1):
        egoNum = i*10
        outName = "policyTrials/ego"+str(egoNum)+'FCFSSwap.csv'
        df, swaps, collisions = firstComefirstServed(egoPolicy, proPolicy, int(vehNum*egoNum/100), int(vehNum*(100-egoNum)/100), False, routeOrder, True, trials)
        if trials == 0:
            df.to_csv(outName)
        else:
            updateCSV(outName, df)
        swapsDict.update({'ego'+str(egoNum): swaps})
        collisionsDict.update({'ego'+str(egoNum): collisions})
    collisionsDF = pd.DataFrame(collisionsDict, index=[0])
    swapsDF = pd.DataFrame(swapsDict, index=[0])
    if trials == 0:
        collisionsDF.to_csv('policyTrials/FCFSCollisions.csv')
        swapsDF.to_csv('policyTrials/swaps.csv')
    else:
        updateCSV('policyTrials/FCFSCollisions.csv', collisionsDF)
        updateCSV('policyTrials/swaps.csv', swapsDF)

    

def readMixedPercentages(min, max, name):
    mixedPolicies = []
    for i in range (min, max+1):
        egoNum = i*10
        fileName = "policyTrials/ego"+str(egoNum)+name+'.csv'
        file = pd.read_csv(fileName)['wait']
        mixedPolicies.append(file)
    return mixedPolicies

def resultsTableRow(arr, name):
    #### OUTPUT: Name | Mean | Standard Deviation | 50th% | 90th% | 95% | 99th%
    row = {'Policy':name, 'Skew':stats.skew(arr), 'Kurtois':stats.kurtosis(arr), 'Mean':np.mean(arr), 'Std':np.std(arr), 'Median':np.percentile(arr,50), '90th':np.percentile(arr, 90), '95th':np.percentile(arr, 95), '99th':np.percentile(arr, 99)}
    return row

def analysisTable(policies, names):
    table = []
    for i in range(len(policies)):
        table.append(resultsTableRow(policies[i], names[i]))
    table = pd.DataFrame(table)
    table.set_index('Policy')
    return table



#GUI INSPECT POLICY
# run(pd.read_csv('policies/egoPolicy20.csv'), None, 'temp/Out.xml', 500, 0, True, None)

##### NEW RESULTS ########
def runPolicies(MaxTrials, vehNumber, egoPolicy, proPolicy):
    trials = 0
    while trials < MaxTrials:
        print('Trial ', str(trials+1), '/', MaxTrials)
        ###### CHOOSE A NEW ROUTE ######
        routeOrder = []
        routeNames = route_creator(False)
        for _ in range(vehNumber):
            routeOrder.append(np.random.choice(routeNames))
        savedRoute = pd.DataFrame({'routeOrder': routeOrder})
        routeName = 'policyTrials/routeOrder'+str(trials)+'.csv'
        savedRoute.to_csv(routeName)

        routeOrderDF = pd.read_csv('policyTrials/routeOrder'+str(trials)+'.csv')
        routeOrder = list(routeOrderDF['routeOrder'])

        ###### UPDATE RESULTS ######
        singlePolicy(egoPolicy, vehNumber, 'policyTrials/egoOUT.csv', routeOrder, False, trials)
        singlePolicy(proPolicy, vehNumber, 'policyTrials/proOUT.csv', routeOrder, False, trials)
        firstComefirstServed(_, _, vehNumber, 0, False, routeOrder, False, trials)
        giveWayToRight(vehNumber, False, routeOrder)
        runMixedPercentages(0, 10, egoPolicy, proPolicy, vehNumber, routeOrder, trials)
        runMixedPercentagesFCFSSwap(0, 10, vehNumber, routeOrder, pd.read_csv(egoPolicy), pd.read_csv(proPolicy), trials)
        trials += 1


vehNumber = 5000
MaxTrials = 3
egoPolicy = 'policies/POLICY_EXP_1_20220423-070405.csv'
proPolicy =  'policies/POLICY_EXP_2_20220422-172212.csv'
runPolicies(MaxTrials, vehNumber, egoPolicy, proPolicy)



###### READ POLICIES
egoPolicyDF = pd.read_csv('policyTrials/egoOUT.csv')['wait']
proPolicyDF = pd.read_csv('policyTrials/proOUT.csv')['wait']
fcfsPolicyDF = pd.read_csv('policyTrials/fcfsOUT.csv')['wait']
gwrPolicyDF = pd.read_csv('policyTrials/gwrOUT.csv')['wait']


# ######## EXPERIMENT 1: EGO vs GWR, FCFS
comparePolicies([egoPolicyDF, gwrPolicyDF, fcfsPolicyDF], ['ego', 'gwr', 'fcfs'])

######### EXPERIMENT 2: PRO vs GWR, FCFS
comparePolicies([proPolicyDF, gwrPolicyDF, fcfsPolicyDF], ['pro', 'gwr', 'fcfs'])

# ######### EXPERIMENT 3: EGO vs PRO
comparePolicies([egoPolicyDF, proPolicyDF], ['ego', 'pro'])
comparePolicies([egoPolicyDF, proPolicyDF, gwrPolicyDF, fcfsPolicyDF], ['ego', 'pro', 'gwr', 'fcfs'])

######## EXPERIMENT 4: MIXED at VARIOUS PERCENTAGES
comparePolicies(readMixedPercentages(0, 10, 'Mixed'),[str(i*10) for i in range(11)]) 
df = pd.read_csv('policyTrials/MixedCollisions.csv')
plotter.plotArr(df.iloc[0][1::], 'Collisions with Percentage of Ego Vehicles', 'Percentage of Ego Vehicles', 'Collisions per 5000 vehicles')
####### EXPERIMENT 5: MIXED OPTIMAL? vs FCFS, GWR
# comparePolicies([ego80mixedPolicyDF, gwrPolicyDF, fcfsPolicyDF], ['optimalPolicy', 'gwr', 'fcfs'])

######## EXPERIMENT 6: FCFSswap AT VARIOUS PERCENTAGES
comparePolicies(readMixedPercentages(0, 10, 'FCFSSwap'),[str(i*10) for i in range(0,11)]) 

######### EXPERIMENT 7: COMPARE FCFSSwap TO FCFS and MIXED 
comparePolicies(readMixedPercentages(4, 8, 'FCFSSwap')+[fcfsPolicyDF,egoPolicyDF,proPolicyDF], [str(i*10) for i in range(4,9)]+['fcfs', 'ego', 'pro'])
df = pd.read_csv('policyTrials/swaps.csv')
plotter.plotArr(df.iloc[0][1::], 'Swaps with Percentage of Ego Vehicles', 'Percentage of Ego Vehicles', 'Swaps per 5000 vehicles')

mixedAnalysis = analysisTable(readMixedPercentages(0, 10, 'Mixed'),[str(i*10) for i in range(0,11)]) 
df = pd.read_csv('policyTrials/MixedCollisions.csv')

mixedAnalysis = analysisTable(readMixedPercentages(0, 10, 'Mixed'),[str(i*10) for i in range(0,11)]) 
df = pd.read_csv('policyTrials/MixedCollisions.csv')
mixedAnalysis['Collisions'] = np.array(df.iloc[0][1::])
print(mixedAnalysis)

fcfsSwapAnalysis = analysisTable(readMixedPercentages(0, 10, 'FCFSSwap'),[str(i*10) for i in range(0,11)]) 
df = pd.read_csv('policyTrials/FCFSCollisions.csv')
fcfsSwapAnalysis['Collisions'] = np.array(df.iloc[0][1::])
df = pd.read_csv('policyTrials/swaps.csv')
fcfsSwapAnalysis['Swaps'] = np.array(df.iloc[0][1::])
print(fcfsSwapAnalysis)