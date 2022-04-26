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

def giveWayToRight(vehNumber, gui=False, routeOrder=None, trials=0):
    sumoBinary = sumolib.checkBinary('sumo')
    if gui:
        sumoBinary = sumolib.checkBinary('sumo-gui')
    traci.start([sumoBinary, "-c", "network/grid.sumocfg", "--tripinfo-output", 'temp/Out.xml', "--no-warnings",])
    routeNames = route_creator()
    addVehicles(routeNames, vehNumber, 0, routeOrder)
    cols = 0
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
        cols += traci.simulation.getCollidingVehiclesNumber()
    traci.close()
    policyWait = getWaitTimes('temp/Out.xml')
    df = pd.DataFrame({'wait':policyWait})
    if trials == 0:
        df.to_csv('policyTrials/gwrOUT.csv')
    else:
        updateCSV('policyTrials/gwrOUT.csv', df)
    return cols

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

    # currentRewA = computeReward(queue[0], actionDict)+a[1]
    # currentRewB = computeReward(queue[1], actionDict)+b[0]

    actionDict.update({queue[0]:'0', queue[1]: '1'})
    swappedRewA = computeReward(queue[0], actionDict)+a[0]
    swappedRewB = computeReward(queue[1], actionDict)+b[1]

    # SWAPPED REWARD = LOOKUP STATE WITH ACTION DICT IN Q TABLE
    # swappedRewA = computeReward(queue[0], actionDict)+a[0]
    # swappedRewB = computeReward(queue[1], actionDict)+b[1]

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
    maxWaitTime = 0
    for veh in sumolib.xml.parse(file, 'tripinfo'):
        wait = float(veh.waitingTime)
        if wait >= maxWaitTime:
            maxWaitTime = wait
        if veh.vaporized == 'teleport':
            wait += maxWaitTime
        waitTimes.append(wait)
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
    return cols

def policyInspect(policyCSV):
    policy = pd.read_csv(policyCSV)
    run(policy, None, 'temp/Out.xml', 100, 0, True)

def comparePolicies(policies, titles, title, plotQ=False, bins=40):
    quantiles = [50, 75, 90, 99]
    analysisString = "{}: Skew: {}, Kurtois Skew: {}, Quantles: {}%: {}, {}%: {}, {}%: {}, {}%: {}"
    for i in range(len(policies)):
        q = np.percentile(policies[i], quantiles)
        #print(analysisString.format(titles[i], stats.skew(policies[i]), stats.kurtosis(policies[i]), quantiles[0], q[0], quantiles[1], q[1], quantiles[2], q[2], quantiles[3], q[3]))
    plotter.compareHistograms(bins, policies, titles, quantiles, plotQ, title)

def runMixedPercentages(min, max, egoPolicy, proPolicy, vehNum, routeOrder, trials=0):
    collisionsDict = {}
    for i in range (min, max+1):
        egoNum = i*10
        outName = "policyTrials/ego"+str(egoNum)+'Mixed.csv'
        df, collisions = mixedPolicy(egoPolicy, proPolicy, int(vehNum*egoNum/100), int(vehNum*(100-egoNum)/100), routeOrder, False, outName, trials)
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
    row = {'Policy':name, 'Skew':stats.skew(arr), 'Kurtosis':stats.kurtosis(arr), 'Mean':np.mean(arr), 'Std':np.std(arr), 'Median':np.percentile(arr,50), '75th':np.percentile(arr, 75), '90th':np.percentile(arr, 90), '99th':np.percentile(arr, 99), '99.9th':np.percentile(arr, 99.9)}
    return row

def analysisTable(policies, names):
    table = []
    for i in range(len(policies)):
        table.append(resultsTableRow(policies[i], names[i]))
    table = pd.DataFrame(table)
    table.set_index('Policy')
    table = table.round(2)
    return table



#GUI INSPECT POLICY
# run(pd.read_csv('policies/egoPolicy20.csv'), None, 'temp/Out.xml', 500, 0, True, None)

##### NEW RESULTS ########
def runPolicies(MaxTrials, vehNumber, egoPolicy, proPolicy):
    trials = 0
    collisions = []
    while trials < MaxTrials:
        print('Trial ', str(trials+1), '/', MaxTrials)
        #New Route Per Trial
        # routeOrder = []
        # routeNames = route_creator(False)
        # for _ in range(vehNumber):
        #     routeOrder.append(np.random.choice(routeNames))
        # savedRoute = pd.DataFrame({'routeOrder': routeOrder})
        # routeName = 'policyTrials/routeOrder'+str(trials)+'.csv'
        # savedRoute.to_csv(routeName)

        routeOrderDF = pd.read_csv('policyTrials/routeOrder'+str(trials)+'.csv')
        routeOrder = list(routeOrderDF['routeOrder'])

        egoCollisions =  singlePolicy(egoPolicy, vehNumber, 'policyTrials/egoOUT.csv', routeOrder, False, trials)
        proCollisions =  singlePolicy(proPolicy, vehNumber, 'policyTrials/proOUT.csv', routeOrder, False, trials)
        # _, _, fcfsCollisions = firstComefirstServed(None, None, vehNumber, 0, False, routeOrder, False, trials)
        # gwrCollisions = giveWayToRight(vehNumber, False, routeOrder, trials)
        runMixedPercentages(0, 10, egoPolicy, proPolicy, vehNumber, routeOrder, trials)
        # runMixedPercentagesFCFSSwap(0, 10, vehNumber, routeOrder, pd.read_csv(egoPolicy), pd.read_csv(proPolicy), trials)
        print(egoCollisions, proCollisions)
        # collisions.append({'ego': egoCollisions, 'pro': proCollisions, 'fcfs': fcfsCollisions, 'gwr': gwrCollisions})
        trials += 1
    # colsDF = pd.DataFrame(collisions)
    # colsDF.to_csv('policyTrials/collisions.csv')


vehNumber = 5000
MaxTrials = 3
egoPolicy = 'policies/POLICY_EXP_1_20220423-070405.csv'
proPolicy =  'policies/POLICY_EXP_2_20220422-172212.csv'
# runPolicies(MaxTrials, vehNumber, egoPolicy, proPolicy)



###### READ POLICIES
egoPolicyDF = pd.read_csv('policyTrials/egoOUT.csv')['wait']
proPolicyDF = pd.read_csv('policyTrials/proOUT.csv')['wait']
fcfsPolicyDF = pd.read_csv('policyTrials/fcfsOUT.csv')['wait']
gwrPolicyDF = pd.read_csv('policyTrials/gwrOUT.csv')['wait']




######## EXPERIMENT 1: EGO vs GWR, FCFS
comparePolicies([egoPolicyDF, gwrPolicyDF, fcfsPolicyDF], ['ego', 'gwr', 'fcfs'], 'Average Waiting Times of Ego Policy against FCFS and GWR', True, 30)

print(analysisTable([egoPolicyDF, proPolicyDF, gwrPolicyDF, fcfsPolicyDF], ['Ego', 'Pro', 'GWR', 'FCFS']))

######### EXPERIMENT 2: PRO vs GWR, FCFS
comparePolicies([proPolicyDF, gwrPolicyDF, fcfsPolicyDF], ['pro', 'gwr', 'fcfs'], 'Average Waiting Times of Pro Policy against FCFS and GWR', True, 30)

# ######### EXPERIMENT 3: EGO vs PRO
comparePolicies([egoPolicyDF, proPolicyDF], ['ego', 'pro'], 'Average Waiting Times of Ego against Pro Policies', True)
comparePolicies([egoPolicyDF, proPolicyDF, gwrPolicyDF, fcfsPolicyDF], ['ego', 'pro', 'gwr', 'fcfs'],  'Average Waiting Times of Ego Policy and Pro Policy against FCFS and GWR', True, 30)
collisions = pd.read_csv('policyTrials/collisions.csv')
print(collisions.mean(0))
plotter.plotBar(np.array(collisions.mean(0)[1::]), ['ego', 'pro', 'gwr', 'fcfs'])

######## EXPERIMENT 4: MIXED at VARIOUS PERCENTAGES
comparePolicies(readMixedPercentages(0, 10, 'Mixed'),[str(i*10) for i in range(11)], 'Average Waiting Times of Heterogenous Societies with varying ratios of Ego and Pro Agents', False, 30) 

mixedAnalysis = analysisTable(readMixedPercentages(0, 10, 'Mixed'),[str(i*10) for i in range(0,11)]) 
df = pd.read_csv('policyTrials/MixedCollisions.csv')
mixedAnalysis['Collisions'] = np.array(df.iloc[0][1::])
print(mixedAnalysis)

plotter.plotXArr([mixedAnalysis['Mean'], mixedAnalysis['Collisions']], ['Wait', 'Collisions'], 'Collisions and Mean Wait with Percentage of Ego Vehicles', 'Percentage of Ego Vehicles', 'Average Collisions and Wait per 15000 vehicles')


####### EXPERIMENT 5: MIXED OPTIMAL? vs FCFS, GWR
comparePolicies(readMixedPercentages(4, 8, 'Mixed')+[gwrPolicyDF, fcfsPolicyDF], [str(i*10) for i in range(4,9)]+['gwr', 'fcfs'], 'Average Waiting Times of Heterogenous Societies against GWR and FCFS policies')

######## EXPERIMENT 6: FCFSswap AT VARIOUS PERCENTAGES
comparePolicies(readMixedPercentages(0, 10, 'FCFSSwap'),[str(i*10) for i in range(0,11)], 'Average Waiting Time of FCFS Swap policy with varying ratios of Ego and Pro Agents') 

fcfsSwapAnalysis = analysisTable(readMixedPercentages(0, 10, 'FCFSSwap'),[str(i*10) for i in range(0,11)]) 
df = pd.read_csv('policyTrials/FCFSCollisions.csv')
fcfsSwapAnalysis['Collisions'] = np.array(df.iloc[0][1::])
df = pd.read_csv('policyTrials/swaps.csv')
fcfsSwapAnalysis['Swaps'] = np.array(df.iloc[0][1::])
print(fcfsSwapAnalysis)
plotter.plotXArr([fcfsSwapAnalysis['Mean'], fcfsSwapAnalysis['Collisions'], fcfsSwapAnalysis['Swaps']], ['Wait', 'Collisions', 'Swap'], 'Average Swaps, Collisions and Wait with Percentage of Ego Vehicles', 'Percentage of Ego Vehicles', 'Average Swaps, Collisions and Wait per 5000 vehicles')

######### EXPERIMENT 7: COMPARE FCFSSwap TO FCFS, GWR 
comparePolicies(readMixedPercentages(7, 7, 'FCFSSwap')+[fcfsPolicyDF], ['70']+['fcfs'], 'Average Waiting Times of FCFS Swap against GWR and FCFS policies', True)
print(analysisTable(readMixedPercentages(0, 10, 'FCFSSwap')+[fcfsPolicyDF], [str(i*10) for i in range(0,11)]+['fcfs']))
######## EXPERIMENT 8: FCFSSwap AGAINST MIXED OPTIMAL
comparePolicies([pd.read_csv('policyTrials/ego50mixed.csv')['wait'], pd.read_csv('policyTrials/ego70FCFSSwap.csv')['wait']], ['Hetero 50% Ego', 'FCFS SWap 70% Ego'], 'Average Waiting Times of Optimal Mixed against Optimal FCFS Swap policies', True)
print(analysisTable([pd.read_csv('policyTrials/ego50mixed.csv')['wait'], pd.read_csv('policyTrials/ego70FCFSSwap.csv')['wait']], ['Hetero 50% Ego', 'FCFS SWap 70% Ego']))
######## EXPERIMENT 9: 1 Ego Vehicle added - any impact maybe compared to 100 prosocial
