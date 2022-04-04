import traci


def computeReward():
    #calculate reward
    vehicles = traci.vehicle.getIDList()
    reward = 0
    collisions = traci.simulation.getCollidingVehiclesNumber()
    teleport = traci.simulation.getStartingTeleportNumber()
    reward += collisions*-5000 + teleport*-2000
    if collisions == 0:
        reward += 100
    reward -= 10
    totalWait = 0
    totalSpeed = 0
    for vehicle in vehicles:
        totalWait += traci.vehicle.getWaitingTime(vehicle)
        totalSpeed += traci.vehicle.getSpeed(vehicle)
    if len(vehicles)>0:
        reward -= totalWait
        reward += totalSpeed
    
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

def epsilonDecay(episode):
    #epsilon = 0.6
    START_DECAY = 0.6

    END_DECAY = 0.1
    epsilon = max(((EPISODES-episode)/EPISODES)*START_DECAY, END_DECAY)
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


def computeRLActions(states, q_table, epsilon):
    actions = {}
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