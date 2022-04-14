from email import policy
import sumolib
import pandas as pd
import traci
from env import route_creator, addVehicles, getLeadersAtJunctions, getLaneLeaders, getStates, doActions
import plotter
import numpy as np
from scipy import stats

def getActions(states, q_table):
    actions = {}
    for agent in states.keys():
        col = q_table[states[agent]]
        if col.idxmax() == col.idxmin():
            action = np.random.choice(["1", "0"])
        else:
            action = col.idxmax()
        actions.update({agent: str(action)})
    return actions

def getWaitTimes(file):
    waitTimes = []
    for veh in sumolib.xml.parse(file, 'tripinfo'):
        waitTimes.append(float(veh.waitingTime))
    return waitTimes


def run(q_table, policyName, vehicleNumber, gui=True, routeOrder=None):
    sumoBinary = sumolib.checkBinary('sumo')
    if gui:
        sumoBinary = sumolib.checkBinary('sumo-gui')
    traci.start([sumoBinary, "-c", "network/grid.sumocfg", "--no-warnings", "--tripinfo-output", policyName])
    routeNames = route_creator()
    
    addVehicles(routeNames, 0, vehicleNumber, routeOrder)
    traci.simulationStep()
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        states = getStates(getLeadersAtJunctions(getLaneLeaders()))
        
        actions = getActions(states, q_table)
        doActions(actions)
    traci.close() 


def waitTimesToCSV(vehicleNumber):
    pNames = ['egoPolicy50','proPolicy10000']
    #policyName = 'POLICY_EXP_1_20220414-045348.csv'
    #policyName = 'POLICY_EXP_2_20220414-074238.csv'
    dfDict = {}
    routeOrder = []
    routeNames = route_creator(False)
    for _ in range(vehicleNumber):
        routeOrder.append(np.random.choice(routeNames))

    for policyName in pNames:
        policy = pd.read_csv('results/'+policyName+'.csv')
        policyOutName =  'results/OUT_'+str(policyName)+'.xms'
        run(policy, policyOutName, vehicleNumber, False, routeOrder)

        waitTimes = getWaitTimes(policyOutName)
        dfDict.update({policyName:waitTimes})

    waitDf = pd.DataFrame(dfDict)
    waitDf.to_csv('waitTimes.csv')

def plotWaitTimes(df):
    
    plotter.compareHistograms(df['egoPolicy50'], df['proPolicy10000'], 'egoPolicy50', 'proPolicy10000')
    #plotter.plotHistogram(df['proPolicy10000'], 'proPolicy10000')

waitTimesToCSV(5000)
df = pd.read_csv('waitTimes.csv')
print('Ego Skew: ', stats.skew(df['egoPolicy50']), 'Kurtois Skew: ', stats.kurtosis(df['egoPolicy50']))
print('Pro Skew: ', stats.skew(df['proPolicy10000']), 'Kurtois Skew: ', stats.kurtosis(df['proPolicy10000']))

plotWaitTimes(df)
