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
import matplotlib as plt


EPISODES = 20
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
    for startEdgeIdx in range(4,8):
        for endEdgeIdx in range(0,4):
            if startEdgeIdx != endEdgeIdx+4:
                startEdge = str(startEdgeIdx)
                endEdge = str(endEdgeIdx)
                name = startEdge+endEdge
                #print("Creating", name, " route")
                traci.route.add(routeID=name, edges=[str(startEdge), str(endEdge)])
                vehName = "rl_"+startEdge+endEdge
                traci.vehicle.add(vehName, name, "car")
                #print("Added: Vehicle", vehName, ", Route ", name)

def DEPRECATEDgetLeadersAtJunctions(leaders, junctionCenters):
    #junctions = traci.junction.getIDList()
    #DEPRECATED: junctionCenters = [traci.junction.getPosition(junction) for junction in junctions] #finds junction centers
    leadersAtJunction, edgesWithLeaders = [], []
    for leader in leaders.values():
        (xPos, yPos) = traci.vehicle.getPosition(leader) 
        for (xJunc, yJunc) in junctionCenters:
            if abs(xPos-xJunc) <= 16 and abs(yPos-yJunc) <= 16:
                traci.vehicle.setSpeed(leader, 0)
                leadersAtJunction.append(leader)
    leadersAtJunction = set(leadersAtJunction)
    return list(leadersAtJunction)

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

def estimateNewState(state, action):
    #if one behing each leader in state and action = 1 then state remains 1 otherwise 0
    stateArr = arrayHelper(state)
    actionArr = arrayHelper(action)
    orderedLanes = ['4_0', '5_0', '6_0', '7_0']
    for i in range(len(stateArr)):
        if stateArr[i] == "1" and actionArr[i] == "1":
            #vehicle present and go
            #get veh_id and get follower and distance and if its small and true then state remains 1 
            vehicleAtjunction = recurisveLaneLeader(orderedLanes[i])
            follower = traci.vehicle.getFollower(vehicleAtjunction, dist=0)
            if follower:
                stateArr[i] = "1"
            else:
                stateArr[i] = "0"
        if stateArr[i] == "1" and actionArr[i] == "0": #agent wont move so state remains 1
            stateArr[i] = "1"
        if stateArr[i] == "0":
            vehicleApproaching = recurisveLaneLeader(orderedLanes[i])
            if vehicleApproaching and traci.vehicle.getLanePosition(vehicleApproaching) > 75:
                stateArr[i] = "1"
            else:
                stateArr[i] = "0"
    return stringHelper(stateArr)


def update_Q_value(state, action, new_state, q_table):
    current_q = q_table[state][action]
    best_predicted_q = q_table[new_state].max()
    new_q = (1-LEARNING_RATE)*current_q + LEARNING_RATE*(computeReward()+DISCOUNT_FACTOR*best_predicted_q)
    q_table[state][action] = new_q
    

def init_Q_table():
    rows = set([x for x in itertools.combinations(["0", "1", "X"]*4, 4)])
    rows = [stringHelper(r) for r in rows]
    columns = set([x for x in itertools.combinations(["0", "1"]*4, 4)])
    columns = [stringHelper(r) for r in columns]
    df = pd.DataFrame(0, columns=columns, index=rows)
    return df


def arrayHelper(actions):
    return [char for char in actions]

def computeRLAction(state, q_table):
    epsilon = epsilonDecay()
    exploreExploit = np.random.random()
    stateString = stringHelper(state)
    if exploreExploit < epsilon: #Choose highest q value from table for this state 
        col = q_table[stateString]
        actions = col.idxmax()
        if actions.count("X") != state.count("0"): #no data on this state so could pick an invalid action
            actions = getRandomActions(state)
    else:
        actions = getRandomActions(state)
    update_Q_value(stateString, actions, estimateNewState(stateString, actions), q_table)
    return actions

def getLeadersAtJunctions(leaders):
    leadersAtJunction = []
    for leader in leaders.values():
            if traci.vehicle.getLanePosition(leader) > 83 and traci.vehicle.getLaneID(leader) in ['4_0', '5_0', '6_0', '7_0']: #trial replacement of LeadersATJunction()
                traci.vehicle.setSpeed(leader, 0)
                leadersAtJunction.append(leader)
    return leadersAtJunction

def recurisveLaneLeader(lane):
    vehiclesOnLane = traci.lane.getLastStepVehicleIDs(lane)
    for vehicle in vehiclesOnLane:
        if traci.vehicle.getLeader(vehicle) == None:     
            return vehicle

def getLaneLeaders():
    lanes = traci.lane.getIDList()
    leaders = {}
    for lane in  ['4_0', '5_0', '6_0', '7_0']:
        leader = recurisveLaneLeader(lane)
        if leader:
            leaders.update({lane:leader})
    return leaders

def getCurrentState(leadersAtJunction):
    state = [0, 0, 0, 0]
    for leader in leadersAtJunction:
        roadId = traci.vehicle.getRoadID(leader)
        if len(roadId)==1: #if not junction and leader
            state[int(roadId)-4] = 1
    return state

def doActions(actions, leadersAtJunction):
    count = 0
    for actionIdx in range(len(actions)):
        if actions[actionIdx] == "1":
            traci.vehicle.setSpeedMode(leadersAtJunction[count], 32)
            traci.vehicle.setSpeed(leadersAtJunction[count], 10)
            count+= 1
        if actions[actionIdx] == "0":
            count += 1


#traCI control loop
def run():
    steps = []
    waitingTime = []
    dataFrame = init_Q_table()
    for episode in range(EPISODES):
        print("EPISODE: ", episode, "/", EPISODES)
        traci.start([
        sumoBinary, "-c", "grid.sumocfg", "--tripinfo-output", "tripinfo.xml"
        ])
        step = 0
        network_creator() #adds vehicles and creates routes
        traci.simulationStep()

        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            
            leaders = getLaneLeaders()
            leadersAtJunction = getLeadersAtJunctions(leaders)
            state = getCurrentState(leadersAtJunction)

            if stringHelper(state) != "0000": #agents waiting at junction
                actions = computeRLAction(state, dataFrame)
                doActions(arrayHelper(actions), leadersAtJunction)
            step += 1
        steps.append(step)
        waitingTime.append(traci.simulation.getTime())
        traci.close()
        sys.stdout.flush()
    #plt.scatter(steps)
    #plt.show()
    print(dataFrame)
    print(steps)
    print(waitingTime)
  

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
