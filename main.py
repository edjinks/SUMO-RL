#!/usr/bin/env python
#netgenerate --g --grid.number=4 -L=1 --grid.length=100 --grid.attach-length 100 --lefthand

from curses.ascii import isdigit
from itertools import count
from operator import indexOf
import sumolib
import os
import sys
import optparse
import numpy as np
import random
import pandas as pd
import itertools
import matplotlib.pyplot as plt


EPISODES = 500
LEARNING_RATE = 0.2
DISCOUNT_FACTOR = 0.95

# we need to import some python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # Checks for the binary in environ vars
import traci

def get_options():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--nogui",
                          action="store_true",
                          default=False,
                          help="run the commandline version of sumo")
    options, args = opt_parser.parse_args()
    return options

def network_creator():
    routeNames = []
    vehNames = []
    for startEdgeIdx in range(4,8):
        for endEdgeIdx in range(0,4):
            if startEdgeIdx != endEdgeIdx+4:
                startEdge = str(startEdgeIdx)
                endEdge = str(endEdgeIdx)
                name = startEdge+endEdge
                #print("Creating", name, " route")
                traci.route.add(routeID=name, edges=[str(startEdge), str(endEdge)])
                routeNames.append(name)
                #traci.vehicle.add(vehName, name, "car")
    for i in range(50):
        vehName = "rl_"+str(i)
        name = np.random.choice(routeNames)
        traci.vehicle.add(vehName, name, "car")
                #print("Added: Vehicle", vehName, ", Route ", name)

def computeReward():
    #calculate reward
    vehicles = traci.vehicle.getIDList()
    reward = 0
    collisions = traci.simulation.getCollidingVehiclesNumber()
    teleport = traci.simulation.getStartingTeleportNumber()
    reward += collisions*-5000 + teleport*-2000
    if collisions == 0:
        reward += 100
    reward -= 1
    totalWait = 0
    totalSpeed = 0
    for vehicle in vehicles:
        totalWait += traci.vehicle.getWaitingTime(vehicle)
        totalSpeed += traci.vehicle.getSpeed(vehicle)
    if len(vehicles)>0:
        reward -= totalWait/len(vehicles)
        reward += totalSpeed/len(vehicles)
    
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

def epsilonDecay():
    epsilon = 0.5
    # START_DECAY = 1
    # END_DECAY = EPISODES//2
    # epsilon = epsilon/(END_DECAY-START_DECAY)
    return epsilon

def estimateNewStates(states, actions):
    leadersAtJunctions = getLeadersAtJunctions(getLaneLeaders())
    estimatedLeadersAtJunction = {}
    agentToOldAgentDict = {}
    #lanes = ['4_0', '5_0', '6_0', '7_0']
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
        #if no actipn for lane check leader wont have arrived
    #maybe not needed if none following as state for it unknown so take someone elses state

    # for i in lanes:
    #     if i not in estimatedLeadersAtJunction.keys():
    #         vehicleApproaching = recurisveLaneLeader(i)

    #         if vehicleApproaching and traci.vehicle.getLanePosition(vehicleApproaching) > 75:
    #             estimatedLeadersAtJunction.update({i:vehicleApproaching})
    # #             estimatedLeadersAtJunction.update({i:vehicleApproaching})
    # #             agentToOldAgentDict.update({leader:vehicleApproaching})
            
    estimatedLeadersAtJunction = {k: v for k, v in estimatedLeadersAtJunction.items() if v}
    newStates = getStates(estimatedLeadersAtJunction)

    oldAgentsNewStates = {}
    for leader in newStates.keys():
        state = newStates[leader]
        oldAgent = agentToOldAgentDict[leader]
        oldAgentsNewStates.update({oldAgent:state})
    return oldAgentsNewStates



def update_Q_value(state, action, new_state, q_table):
    current_q = q_table[state][action]
    best_predicted_q = q_table[new_state].max()
    new_q = (1-LEARNING_RATE)*current_q + LEARNING_RATE*(computeReward()+DISCOUNT_FACTOR*best_predicted_q)
    q_table[state][action] = new_q
    return q_table
    

def init_Q_table():
    rows = ["0", "1"]
    columns = set([i for i in itertools.permutations(["0","1","2","3","4"]*4, 4)])
    columns = [stringHelper(r) for r in columns]
    df = pd.DataFrame(0, columns=columns, index=rows)
    return df


def arrayHelper(actions):
    return [char for char in actions]

def getLeadersAtJunctions(leaders):
    leadersAtJunction = {}
    for lane in leaders:
            leader = leaders[lane]
            if traci.vehicle.getLanePosition(leader) > 83 and traci.vehicle.getLaneID(leader) in ['4_0', '5_0', '6_0', '7_0']: #trial replacement of LeadersATJunction()
                traci.vehicle.setSpeed(leader, 0)
                leadersAtJunction.update({lane:leader})
    return leadersAtJunction

def recurisveLaneLeader(lane):
    vehiclesOnLane = traci.lane.getLastStepVehicleIDs(lane)
    for vehicle in vehiclesOnLane:
        if traci.vehicle.getLeader(vehicle) == None:     
            return vehicle

def getLaneLeaders():
    #lanes = traci.lane.getIDList()
    leaders = {}
    for lane in  ['4_0', '5_0', '6_0', '7_0']:
        leader = recurisveLaneLeader(lane)
        if leader:
            leaders.update({lane:leader})
    return leaders

def computeRLActions(states, q_table):
    actions = {}
    epsilon = epsilonDecay()

    for agent in states.keys():
        exploreExploit = np.random.random()
        if exploreExploit < epsilon: #Choose highest q value from table for this state 
            col = q_table[states[agent]]
            action = col.idxmax()
        else:
            action = np.random.choice(["1", "0"])
        actions.update({agent: action})
    if len(actions.items()) != 0:
        new_states = estimateNewStates(states, actions)
        for agent in states.keys():
            if agent not in new_states.keys():
                new_state = states[agent]
            else:
                new_state = new_states[agent]
            q_table = update_Q_value(states[agent], action, new_state, q_table)    
    return actions, q_table

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
    #print(states)
    return states




def doActions(actions):
    for veh in actions.keys():
        if actions[veh] == "1":
            traci.vehicle.setSpeedMode(veh, 32)
            traci.vehicle.setSpeed(veh, 10)

def pltResults(steps, waitingTime):
    x = np.array([x for x in range(0,500)])

    y = np.array(steps)
    y2 = np.array(waitingTime)
    print(len(x), len(y), len(y2))
    m, b = np.polyfit(x, y, 1)
    m2, b2 = np.polyfit(x, y2, 1)
    plt.figure(figsize=(20, 5))
    plt.plot(x, y, 'o')
    plt.plot(x, y2, 'o')
    plt.plot(x, m*x+b)
    plt.plot(x, m2*x+b2)
    plt.tight_layout()
    plt.show()


#traCI control loop
def run():
    steps = []
    waitingTime = []
    dataFrame = init_Q_table()
    print(dataFrame)
    for episode in range(EPISODES):
        print("EPISODE: ", episode, "/", EPISODES)
        traci.start([
        sumoBinary, "-c", "grid.sumocfg", "--tripinfo-output", "tripinfo.xml", "--no-warnings"
        ])
        step = 0
        network_creator() #adds vehicles and creates routes
        traci.simulationStep()

        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            leaders = getLaneLeaders()
            leadersAtJunction = getLeadersAtJunctions(leaders)
            states = getStates(leadersAtJunction)
            actions, dataFrame = computeRLActions(states, dataFrame)
            doActions(actions)
            step += 1
        steps.append(step)
        waitingTime.append(traci.simulation.getTime())
        traci.close()
        sys.stdout.flush()
    print(dataFrame)
    print(steps)
    print(waitingTime)
    pltResults(steps, waitingTime)
  

# main entry point
if __name__ == "__main__":
    options = get_options()
    # check binary
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')
    # traci starts sumo as a subprocess and then this script connects and runs
    # traci.start([
    #     sumoBinary, "-c", "grid.sumocfg", "--tripinfo-output", "tripinfo.xml"
    # ])
    run()
