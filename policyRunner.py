#!/usr/bin/env python
#netgenerate --g --grid.number=4 -L=1 --grid.length=100 --grid.attach-length 100 --lefthand
from sumolib import checkBinary
import numpy as np
import pandas as pd
import itertools
import traci


from env import route_creator, addVehicles, getLeadersAtJunctions, getLaneLeaders, getStates, doActions


def getActions(states, q_table):
    actions = {}
    for agent in states.keys():
        col = q_table[states[agent]]
        action = col.idxmax()
        actions.update({agent: action})
    return actions


def run(q_table, gui=True):
    sumoBinary = checkBinary('sumo')
    if gui:
        sumoBinary = checkBinary('sumo-gui')
    traci.start([sumoBinary, "-c", "network/grid.sumocfg", "--no-warnings"])
    addVehicles(route_creator(), 100, 0)
    traci.simulationStep()
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        states = getStates(getLeadersAtJunctions(getLaneLeaders()))
        actions = getActions(states, q_table)
        doActions(actions)
    traci.close() 



df = pd.read_csv('results/experiment2.csv')

run(df)