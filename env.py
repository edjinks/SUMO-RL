#!/usr/bin/env python
#GENERATE NETWORK COMMAND FOR NETEDIT
#netgenerate --g --grid.number=4 -L=1 --grid.length=100 --grid.attach-length 100 --lefthand

import numpy as np
import pandas as pd
import itertools
import traci
import sumolib
from sumolib import checkBinary
import random

def route_creator(add=True):
    routeNames = []
    for startEdgeIdx in range(4,8):
        for endEdgeIdx in range(0,4):
            if startEdgeIdx != endEdgeIdx+4:
                startEdge = str(startEdgeIdx)
                endEdge = str(endEdgeIdx)
                name = startEdge+endEdge
                if add:
                    traci.route.add(routeID=name, edges=[str(startEdge), str(endEdge)])
                routeNames.append(name)
    return routeNames

def addVehicles(routeNames, numEgoists, numProsocialists, routeOrder = None, vehOrder=None, add=True):
    if vehOrder is None:
        vehNames = []
        for i in range(numEgoists):
            vehNames.append("EGO_vl"+str(i))
        for j in range(numProsocialists):
            vehNames.append("PRO_vl"+str(j))
        random.shuffle(vehNames)
    else:
        vehNames = vehOrder
    for j in range(len(vehNames)):
        if routeOrder is not None:
            routeName = str(routeOrder[j])
        else:
            routeName = np.random.choice(routeNames)
        stopEdge = str(list(routeName)[0])
        if add:
            traci.vehicle.add(vehNames[j], routeName, "car")
            traci.vehicle.setStop(vehNames[j], edgeID=stopEdge, laneIndex=0, pos=90, duration=1000.0) #stops for 1 step at junction to make decision/detect being leader at junction
    return vehNames

def computeEgotistReward(veh_id, action):
    reward = 0
    ownWait = traci.vehicle.getAccumulatedWaitingTime(veh_id)
    reward -= ownWait
    if action == '1':
         reward += 100
    collisionList = traci.simulation.getCollidingVehiclesIDList()
    if veh_id in collisionList:
        reward -= 500
    return reward

def computeProsocialReward(actions):
    reward = 0
    collisionNumber = traci.simulation.getCollidingVehiclesNumber()
    if collisionNumber == 0:
        reward += 10
    reward = reward*sum([int(a) for a in actions.values()])
    reward -= 500*collisionNumber
    return reward

def computeReward(agent, actions):
    if 'EGO' in agent:
        reward = computeEgotistReward(agent, actions[agent])
    elif 'PRO' in agent:
        reward = computeProsocialReward(actions)
    return reward

def stringHelper(state):
    stateString = ""
    for i in state:
        stateString += str(i)
    return stateString

def getRandomActions(state):
    actions = ["X", "X", "X", "X"]
    for i in range(4):
        if state[i] == 1:
            actions[i] = np.random.choice([0,1])
    return stringHelper(actions)

def epsilonDecay(params, episode):
    epsilon = max(((params['EPISODES']-episode)/params['EPISODES'])*params['START_EPSILON_DECAY'], params['END_EPSILON_DECAY'])
    return epsilon

def estimateNewStates(actions):
    leadersAtJunctions = getLeadersAtJunctions(getLaneLeaders())
    estimatedLeadersAtJunction = {}
    agentToOldAgentDict = {}
    for lane in leadersAtJunctions.keys():
        leader = leadersAtJunctions[lane]
        if actions[leader] == "1":
            vehicleAtjunction = recurisveLaneLeader(lane)
            follower = traci.vehicle.getFollower(vehicleAtjunction, dist=0)
            if follower:
                estimatedLeadersAtJunction.update({lane:follower[0]})
                agentToOldAgentDict.update({follower[0]:leader})
        if actions[leader] == "0":
            estimatedLeadersAtJunction.update({lane:leader})
            agentToOldAgentDict.update({leader:leader})
    estimatedLeadersAtJunction = {k: v for k, v in estimatedLeadersAtJunction.items() if v}
    newStates = getStates(estimatedLeadersAtJunction)

    oldAgentsNewStates = {}
    for leader in newStates.keys():
        state = newStates[leader]
        oldAgent = agentToOldAgentDict[leader]
        oldAgentsNewStates.update({oldAgent:state})
    return oldAgentsNewStates

def update_Q_value(agent, state, actions, new_state, q_table, params, rewards,epsilon):
    action = actions[agent]
    current_q = q_table[state][action]
    best_predicted_q = q_table[new_state].max()
    LEARNING_RATE = params['LEARNING_RATE']
    reward = computeReward(agent, actions)
    rewards += reward
    q_table[state][action] = (1-LEARNING_RATE)*current_q + LEARNING_RATE*epsilon*(reward+params['DISCOUNT_FACTOR']*best_predicted_q)
    return q_table, rewards

def init_Q_table():
    rows = ["0", "1"]
    states = list(itertools.product(*[[1,2,3], [0,2,3,4], [0,1,3,4], [0,1,2,4]]))
    columns = [''.join([str(i) for i in state]) for state in states]
    df = pd.DataFrame(0, columns=columns, index=rows)
    return df

def getLeadersAtJunctions(leaders):
    leadersAtJunction = {}
    for lane in leaders:
            leader = leaders[lane]
            if traci.vehicle.isStopped(leader):
                leadersAtJunction.update({lane:leader})
    return leadersAtJunction

def recurisveLaneLeader(lane):
    vehiclesOnLane = traci.lane.getLastStepVehicleIDs(lane)
    for vehicle in vehiclesOnLane:
        if traci.vehicle.getLeader(vehicle) == None:     
            return vehicle

def getLaneLeaders():
    leaders = {}
    for lane in  ['4_0', '5_0', '7_0', '6_0']:
        leader = recurisveLaneLeader(lane)
        if leader:
            leaders.update({lane:leader})
    return leaders

def computeRLActions(states, q_table, epsilon):
    actions = {}
    for agent in states.keys():
        exploreExploit = np.random.random()
        if exploreExploit > epsilon: #Choose highest q value from table for this state 
            col = q_table[states[agent]]
            action = col.idxmax()
        else:
            action = np.random.choice(["1", "0"])
        actions.update({agent: action})
    return actions

def updateTable(actions, q_table, states, new_states, params, rewards, epsilon):
    if len(actions.items()) != 0:
        for agent in states.keys():
            if agent not in new_states.keys():
                new_state = states[agent]
            else:
                new_state = new_states[agent]
            q_table, rewards = update_Q_value(agent, states[agent], actions, new_state, q_table, params, rewards, epsilon)    
    return q_table, rewards

def getStates(leadersAtJunction):
    states = {}
    for lane in leadersAtJunction.keys():
        state = ["0", "0", "0", "0"]
        if lane == "7_0": #north approach
            destLaneDict = {"2_0": "1", "0_0": "2", "1_0": "3", "3_0": "4"}
            left = "6_0"
            right = "5_0"
            straight = "4_0"
        if lane == "6_0": #east
            destLaneDict = {"0_0": "1", "1_0": "2", "3_0": "3", "2_0": "4"}
            left = "4_0"
            right = "7_0"
            straight = "5_0"
        if lane == "4_0": #south
            destLaneDict = {"1_0": "1", "3_0": "2", "2_0": "3", "0_0": "4"}
            left = "5_0"
            right = "6_0"
            straight = "7_0"
        if lane == "5_0": #west
            destLaneDict = {"3_0": "1", "2_0": "2", "0_0": "3", "1_0": "4"}
            left = "7_0"
            right = "4_0"
            straight = "6_0"

        #OWN DESTINATION state[0]            
        route = traci.vehicle.getRoute(leadersAtJunction[lane])
        destLane = str(route[1])+"_0"
        number = destLaneDict[destLane]
        state[0] = number
        if left in leadersAtJunction.keys():
            route = traci.vehicle.getRoute(leadersAtJunction[left])
            destLane = str(route[1])+"_0"
            number = destLaneDict[destLane]
            state[1] = number
        if straight in leadersAtJunction.keys():
            route = traci.vehicle.getRoute(leadersAtJunction[straight])
            destLane = str(route[1])+"_0"
            number = destLaneDict[destLane]
            state[2] = number
        if right in leadersAtJunction.keys():
            route = traci.vehicle.getRoute(leadersAtJunction[right])
            destLane = str(route[1])+"_0"
            number = destLaneDict[destLane]
            state[3] = number
       
        states.update({leadersAtJunction[lane]:stringHelper(state)})
    return states

def doActions(actions):
    for veh in actions.keys():
        if actions[veh] == '1':
            traci.vehicle.resume(veh)
            traci.vehicle.setSpeedMode(veh, 32)

def outputParse(outFilename):
    totalWaitTime = 0
    count = 0
    for v in sumolib.xml.parse(outFilename, 'tripinfo'):
        count += 1
        totalWaitTime += float(v.waitingTime)
    return totalWaitTime/count


def run(params, gui=False):
    sumoBinary = checkBinary('sumo')
    if gui:
        sumoBinary = checkBinary('sumo-gui')

    steps = []
    episodicRewards = []
    episodeAvgWaitCount = []
    dataFrame = init_Q_table()

    for episode in range(int(params['EPISODES'])):
        rewards = 0
        outFileName = 'temp/out_'+str(params['EGOISTS'])+'.xml'
        traci.start([sumoBinary, "-c", "network/grid.sumocfg", "--tripinfo-output", outFileName, "--no-warnings"])
        epsilon = epsilonDecay(params, episode)
        print("EPISODE: ", episode+1, "/", params['EPISODES'], " EPSILON: ", epsilon, " LEARNING RATE: ", params['LEARNING_RATE'])
        
        step = 0
        actions = None
        states = None
        addVehicles(route_creator(), params['EGOISTS'], params['PROSOCIALISTS'])
        traci.simulationStep()
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            traci.simulationStep()
            if actions and states:
                dataFrame, rewards = updateTable(actions, dataFrame, states, new_states, params, rewards, epsilon)
            states = getStates(getLeadersAtJunctions(getLaneLeaders()))
            actions = computeRLActions(states, dataFrame, epsilon)
            if len(actions) != 0:
                new_states = estimateNewStates(actions)
            doActions(actions)
            step += 1
        averageReward = rewards/(params['EGOISTS']+params['PROSOCIALISTS'])
        episodicRewards.append(averageReward)
        steps.append(step)
        traci.close() 
        episodeAvgWaitCount.append(outputParse(outFileName))
    return dataFrame, episodicRewards, steps, episodeAvgWaitCount
  