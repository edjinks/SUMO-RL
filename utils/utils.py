
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


def pltResults(steps, waitingTime):
    x = np.array([x for x in range(0,len(waitingTime))])
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

def pltCummulativeReward(cummulativeReward):
    plt.plot(cummulativeReward)
    plt.show()
