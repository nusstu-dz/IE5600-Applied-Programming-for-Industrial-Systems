import copy
import os
import math
import numpy as np
import random


class Node(object):
    def __init__(self, idx, x, y, load, minTime, maxTime):
        super(Node, self).__init__()
        self.idx = idx
        self.x = x
        self.y = y
        self.load = load
        self.minTime = minTime
        self.maxTime = maxTime
        self.profit = 0


def getDist(location1, location2):
    x1 = allNodes[location1].x
    y1 = allNodes[location1].y
    x2 = allNodes[location2].x
    y2 = allNodes[location2].y
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def travelTime(startTime, startNode, endNode):
    # infeasible
    if startTime >= Tmax:
        return False

    arcCat = speedChooseMat[startNode][endNode]
    speed = speedMat[arcCat]
    distance = distMat[startNode][endNode]

    # determine start in which timezone
    for i in range(len(speed)):
        if startTime >= speed[i][0] and startTime < speed[i][1]:
            timezone = i
            break

    # calculate time taken
    timeTravelled = 0
    for i in range(timezone, len(speed)):
        maxDistance = (speed[i][1] - startTime) * speed[i][2]

        # if cannot reach destination in this timezone, check next timezone
        if distance > maxDistance:
            distance -= maxDistance
            timeTravelled += speed[i][1] - startTime

        # can reach in this timezone
        else:
            timeTravelled += distance / speed[i][2]
            distance = 0
            break
        startTime = speed[i][1]

    if distance == 0:
        return (timeTravelled)

    # cannot reach within the last timezone == infeasible
    else:
        return False


def checkFeasible(path, timing=False): #timing=True to return time taken of the route
    
    if path == [0,0]:
        return 0.0

    # check load capacity
    load = 0
    for i in range(len(path) - 1):
        nextNode = path[i + 1]
        load += allNodes[nextNode].load
        if load > maxCapacity:
            return False

    # check time
    startNode = path[0]
    firstNode = path[1]
    #get latest starting time to leave depot
    if allNodes[firstNode].minTime > travelTime(0,startNode,firstNode):
        startTime = allNodes[firstNode].minTime - travelTime(0,startNode,firstNode)
    else: 
        startTime = 0

    #calculate time to reach back to the depot
    time = 0
    profit = 0
    for i in range(len(path) - 1):
        currNode = path[i]
        nextNode = path[i + 1]
        timeTaken = travelTime(time, currNode, nextNode)
        if timeTaken and time < allNodes[nextNode].maxTime:
            time = max(time + timeTaken, allNodes[nextNode].minTime)
        else:
            return False
        profit += allNodes[nextNode].profit

    #pass all checks, calculate profit
    objFunction = profit - time+startTime

    if timing:
        return time

    if objFunction > 0:
        return objFunction
    else: 
        return False

def BFS():
    basicSolution = [[0,0] for _ in range(noOfVehicles)]

    remainingPickups = copy.deepcopy(pickupNodes)
    currOptimalSolution = []
    for vehicle in basicSolution:
        if remainingPickups != {}:
            #initialse the next best pickup
            bestTiming = float('inf')
            pickupFlag = False
            #insert earliest pickup option to allow for more pickups later
            for item in remainingPickups:
                testPath = copy.deepcopy(vehicle)
                testPath.insert(-1, item)
                testPath.insert(-1, pickupDeliveryPair[item])
                timing = checkFeasible(testPath, timing=True)
                score = checkFeasible(testPath)
                if timing and score and timing < bestTiming:
                    bestTiming = timing
                    bestPath = testPath
                    pickupFlag = True
            if pickupFlag: #check if there is a initial feasible pickup
                vehicle = bestPath
                remainingPickups.pop(vehicle[-3]) #remove last route taken


            #insert the rest based on objectiveFunction
            bestScore = checkFeasible(vehicle)
            availablePaths = True

            while availablePaths:
                availablePaths = False
                for item in remainingPickups:
                    testPath = copy.deepcopy(vehicle)
                    testPath.insert(-1, item)   #insert pickup node
                    testPath.insert(-1, pickupDeliveryPair[item])   #insert delivery node

                    score = checkFeasible(testPath)
                    if score and score > bestScore:
                        availablePaths = True
                        bestScore = score
                        bestPath = testPath
                if availablePaths:
                    vehicle = bestPath
                    remainingPickups.pop(vehicle[-3]) #remove last route taken

        currOptimalSolution.append(vehicle)  
    return currOptimalSolution

def RandomBFS():
    basicSolution = [[0, 0] for _ in range(noOfVehicles)]

    remainingPickups = copy.deepcopy(pickupNodes)
    currOptimalSolution = []
    counter = 0
    while remainingPickups != {}:

        # initialse the next best pickup
        bestTiming = float('inf')
        pickupFlag = False
        # insert earliest pickup option to allow for more pickups later
        car = np.random.randint(len(basicSolution))
        vehicle = basicSolution[car]
        for item in remainingPickups:

            testPath = copy.deepcopy(vehicle)
            testPath.insert(-1, item)
            testPath.insert(-1, pickupDeliveryPair[item])
            timing = checkFeasible(testPath, timing=True)
            score = checkFeasible(testPath)
            counter += 1
            if timing and score and timing < bestTiming:
                bestTiming = timing
                bestPath = testPath
                pickupFlag = True

        if pickupFlag:  # check if there is a initial feasible pickup
            vehicle = bestPath
            remainingPickups.pop(vehicle[-3])  # remove last route taken
            basicSolution[car] = vehicle

        if counter > 10000:
            break

    currOptimalSolution.append(vehicle)
    return basicSolution


'''inputs'''
os.chdir('input/')
questionInput = open('Prob-30A-50.txt', 'r')
questionInput = questionInput.readlines()

noOfVehicles = int(questionInput[0])
maxCapacity = int(questionInput[1])
Tmax = int(questionInput[2])

depot = questionInput[5].replace(',', '.').split()
depot = Node(int(depot[0]), float(depot[1]), float(depot[2]), 0, 0, Tmax)

pickupNodes = {}
requests=0

for i in range(9, 999):
    # additional logic to detect end of pick up nodes
    if len(questionInput[i]) < 3:
        break
    else:
        node = questionInput[i].replace(',', '.').split()
        if node==[]:
            break
        pickupNodes[int(node[0])] = Node(int(node[0]), float(node[1]), float(node[2]), int(node[4]), float(node[6]),
                                     float(node[7]))
        # count number of requests
        requests += 1

deliveryNodes = {}
for i in range(9+requests+3, 9+requests+3+requests):
    node = questionInput[i].replace(',', '.').split()
    deliveryNodes[int(node[0])] = Node(int(node[0]), float(node[1]), float(node[2]), int(node[4]), float(node[6]),
                                       float(node[7]))
    deliveryNodes[int(node[0])].profit = 80  #each node's profit upon delivery

allNodes = {0: depot, **pickupNodes, **deliveryNodes}

# build the pickup delivery matching dict
pickupDeliveryPair = {}
iter = 1
for item in deliveryNodes:
    pickupDeliveryPair[iter] = deliveryNodes[item].idx
    iter += 1

speedMat = []
# blockcount = 9+requests+3+requests+2 brings you to the first speed pattern in input file
blockcount = 9+requests+3+requests+2
for i in range(5):
    speed = []
    for j in range(i * 6 + blockcount, i * 6 + (blockcount+4)):
        time = questionInput[j].replace(',', '.').split()
        speed.append([float(time[0]), float(time[1]), float(time[3])])
    speedMat.append(speed)

speedChooseMat = []
# use blockcocunt to read the speed choose matrix
for i in range(blockcount+31, blockcount+31+2*requests+1):
    speedChooseMat.append([int(i) for i in questionInput[i].replace(',', '.').split()])

''' processing input '''

# calculate distance matrix
# total lines required = 2*requests + 2 (but we minus 1 because range starts from 0)
distMat = [[0.0] * (2*requests + 2-1) for i in range(2*requests + 2-1)]
for i in range((2*requests + 2-1)):
    for j in range(i + 1, (2*requests + 2-1)):
        dist = getDist(i, j)
        distMat[i][j] = dist
        distMat[j][i] = dist

def Profit_Solution(s1):
    Totaltime=0
    for k_ in range(len(s1)):
        s1[k_][1] = checkFeasible(s1[k_][0])
        # print(s1[k_][1])
        if s1[k_][1] is not False and s1[k_][1] >= 0:
            Totaltime += s1[k_][1]
        elif s1[k_][1] is False:
            Totaltime = 0
            break
    return Totaltime
''' generate BFS '''

currOptimalSolution = RandomBFS()


''' how to use checking function '''

# currOptimalSolution = [
#     [0,1,11,3,13,0],
#     [0,10,20,0],
#     [0,5,15,6,16,0]] #just an example

# #iterate based on this

# print(currOptimalSolution)

s0=[]
for vehicle in currOptimalSolution:
    oneroute=[vehicle, checkFeasible(vehicle)]
    s0.append(oneroute)

nn= len(pickupNodes)
profit0 = Profit_Solution(s0)
print('*******  inital solution')
print(profit0)
print(*s0, sep = "\n")
# print(*s0)
print('********  inital solution')


# print( 'score:' + str(checkFeasible(vehicle)) + '   route:' + str(vehicle)    )


def Remove_node(s, tour, node_pos, data=None):  # data defaulted as none because data struc not in use
    #     print(s[tour])

    # print('Remove ' + str(s[tour][0][node_pos])+ ' at ' + str(node_pos) +' from tour'+ str(tour))
    del s[tour][0][node_pos]


def Insert_node(s, node, tour, point):
    # node_1 = s[tour][point - 1] # Original code uses s[tour][0]

    # arr.insert(point, node)
    # print('Insert ' + str(node) + ' at ' + str(point)+' from tour'+ str(tour))
    s[tour][0].insert(point, node)


# s0 = [[0, 14, 29, 11, 26, 1, 16, 3, 18, 0], 138.45952332463185], [[0, 10, 25, 0], 18.873851271071658], [
#     [0, 8, 23, 6, 21, 13, 28, 5, 20, 15, 30, 4, 19, 2, 17, 0], 366.359325517866], [[0, 7, 22, 0], 29.4378156146646], [
#          [0, 0], False]


def NEIGHBOR_VRP(s, halfnumberofnode, data=None):
    # 1. random pick a tour1

    # for i_ in range(4):
    n = halfnumberofnode
    tour_a = np.random.randint(len(s))
    while len(s[tour_a][0]) <= 4:
        tour_a = np.random.randint(len(s))

    # print (tour_a)

    pos_a = np.random.randint(len(s[tour_a][0]) - 2) + 1  # random int from [1, n-2], n-1 is the last node- depot

    # 2. random pick a tour 2
    tour_b = np.random.randint(len(s))
    # print(tour_b)

    if tour_a == tour_b and len(s[tour_a][0]) >= 6:  # at least two pairs
        point_b = pos_a
        while point_b == pos_a:
            point_b = np.random.randint(len(s[tour_a][0]) - 1) + 1

            # remove node_a
        sNew = copy.deepcopy(s)
        temp_node = sNew[tour_a][0][pos_a]

        ### need detail code here!
        # 2.1  if sNew[tour_a][0][pos_a] is a delivery code: (temp_node>n)

        if sNew[tour_a][0][pos_a] > n:
            pickup_node = sNew[tour_a][0][pos_a] - n
            pos_c = sNew[tour_a][0].index(pickup_node)
            # print(sNew[tour_a][0][pos_a])
            # print(pickup_node)
            sNew[tour_a][0].remove(sNew[tour_a][0][pos_a])
            sNew[tour_a][0].remove(pickup_node)
            # Remove_node(sNew, tour_a, pos_a, data)
            # Remove_node(sNew, tour_a, pos_c, data)
            # make sure the sNew is updated after deleting two nodes

            point_b = np.random.randint(len(sNew[tour_b][0]) - 1) + 1
            point_c = np.random.randint(len(sNew[tour_b][0]) - 1) + 1
            while point_c > point_b:
                point_c = np.random.randint(len(sNew[tour_b][0]) - 1) + 1
            Insert_node(sNew, temp_node, tour_a, point_b)
            Insert_node(sNew, pickup_node, tour_a, point_c)
            return sNew

        # 3.2 if sNew[tour_a][0][pos_a] is a pickup node: (temp_node<=n)
        elif sNew[tour_a][0][pos_a] <= n:
            delivery_node = sNew[tour_a][0][pos_a] + n
            pos_c = sNew[tour_a][0].index(delivery_node)
            # print(sNew[tour_a][0][pos_a])
            # print(delivery_node)
            sNew[tour_a][0].remove(sNew[tour_a][0][pos_a])
            sNew[tour_a][0].remove(delivery_node)
            # Remove_node(sNew, tour_a, pos_c, data)
            # Remove_node(sNew, tour_a, pos_a, data)

            point_b = np.random.randint(len(sNew[tour_b][0]) - 1) + 1
            Insert_node(sNew, temp_node, tour_b, point_b)
            point_c = np.random.randint(len(sNew[tour_a][0]) - 1) + 1
            while point_c <= point_b:
                point_c = np.random.randint(len(sNew[tour_b][0]) - 1) + 1
            Insert_node(sNew, delivery_node, tour_a, point_c)
            return sNew



    # 3. if tour1 != tour 2:

    elif tour_a != tour_b:  # and len(s[tour_a][0]) >=6 : # at least two pairs
        point_b = pos_a
        while point_b == pos_a:
            point_b = np.random.randint(len(s[tour_a][0]) - 1) + 1

            # remove node_a
        sNew = copy.deepcopy(s)
        temp_node = sNew[tour_a][0][pos_a]

        ### need detail code here!
        # 3.1  if sNew[tour_a][0][pos_a] is a delivery code: (temp_node>n)

        if sNew[tour_a][0][pos_a] > n:
            pickup_node = sNew[tour_a][0][pos_a] - n
            pos_c = sNew[tour_a][0].index(pickup_node)
            # print(sNew[tour_a][0][pos_a])
            # print(pickup_node)
            sNew[tour_a][0].remove(sNew[tour_a][0][pos_a])
            sNew[tour_a][0].remove(pickup_node)
            # Remove_node(sNew, tour_a, pos_a, data)
            # Remove_node(sNew, tour_a, pos_c, data)
            # make sure the sNew is updated after deleting two nodes

            point_b = np.random.randint(len(sNew[tour_b][0]) - 1) + 1
            point_c = np.random.randint(len(sNew[tour_b][0]) - 1) + 1
            while point_c > point_b:
                point_c = np.random.randint(len(sNew[tour_b][0]) - 1) + 1
            Insert_node(sNew, temp_node, tour_b, point_b)
            Insert_node(sNew, pickup_node, tour_b, point_c)
            return sNew

        # 3.2 if sNew[tour_a][0][pos_a] is a pickup node: (temp_node<=n)
        elif sNew[tour_a][0][pos_a] <= n:
            delivery_node = sNew[tour_a][0][pos_a] + n
            pos_c = sNew[tour_a][0].index(delivery_node)

            # print(sNew[tour_a][0][pos_a])
            # print(delivery_node)
            sNew[tour_a][0].remove(sNew[tour_a][0][pos_a])
            sNew[tour_a][0].remove(delivery_node)
            # Remove_node(sNew, tour_a, pos_c, data)
            # Remove_node(sNew, tour_a, pos_a, data)

            point_b = np.random.randint(len(sNew[tour_b][0]) - 1) + 1
            Insert_node(sNew, temp_node, tour_b, point_b)
            point_c = np.random.randint(len(sNew[tour_b][0]) - 1) + 1
            while point_c <= point_b:
                point_c = np.random.randint(len(sNew[tour_b][0]) - 1) + 1
            Insert_node(sNew, delivery_node, tour_b, point_c)
            return sNew


# nn=15
# s1= NEIGHBOR_VRP(s0,nn)
# print(s1)




def get_population(S0, halfnodes=nn):
    candidate =[]
    for n_ in range(20):
        s1 = NEIGHBOR_VRP(S0, halfnodes)
        profit = Profit_Solution(s1)
        routes= []
        # for solu in s1:
        #     # solu[0].pop(0)
        #     # solu[0].pop()
        #     route = solu[0]
        #     # print(route)
        #     routes += route
        # routes.insert(0,0)
        # routes.append(0)
        candidate.append((s1, profit))
    candidate.sort(key=lambda x: x[1], reverse=True)
    res= [x[0] for x in candidate]
    return res

#GA Part

def selection(candidate):
    # retain_rate = 0.3
    # graded= candidate.sort(key= lambda x: x[1], reverse=True)
    # graded = [x[0] for x in candidate]
    graded = [x for x in candidate]
    retain_length = int(len(graded) * retain_rate)
    parents = graded[:retain_length]
    for chromosome in graded[retain_length:]:
        if random.random() < random_select_rate:
            parents.append(chromosome)
    return parents

def rank_route(s):
    # routes=[]
    for route in s:
        temp=[]
        route=route.sort(key=lambda x: x[1],reverse=True)
        # route= [i[0] for i in route]
    #     for i in route:
    #         temp.append(route[0])
    # routes.append(temp)
    return s


def remove_profit(s):
    S=[]
    for i in s:
        J=[]
        for j in i:
            J.append(j[0])
        S.append(J)
    return S


def Crossover(parents):
    count=20
    target_count = count - len(parents)
    # 孩子列表
    children = []
    while len(children) < target_count:
        male_index = random.randint(0, len(parents) - 1)
        female_index = random.randint(0, len(parents) - 1)
        if male_index != female_index:
            male = parents[male_index]
            female = parents[female_index]

            left = random.randint(0, len(male) - 2)
            right = random.randint(left + 1, len(male) - 1)

            # 交叉片段
            gene1 = male[left:right]
            gene2 = female[left:right]

            child1_c = male[right:] + male[:right]
            child2_c = female[right:] + female[:right]
            child1 = child1_c.copy()
            child2 = child2_c.copy()

            for o in gene2:
                child1_c.remove(o)

            for o in gene1:
                child2_c.remove(o)

            child1[left:right] = gene2
            child2[left:right] = gene1

            child1[right:] = child1_c[0:len(child1) - right]
            child1[:left] = child1_c[len(child1) - right:]

            child2[right:] = child2_c[0:len(child1) - right]
            child2[:left] = child2_c[len(child1) - right:]

            children.append(child1)
            children.append(child2)

    return children

def crossover(parents):
    count=20
    target_count = count - len(parents)
    # 孩子列表
    children = []
    while len(children) < target_count:
        male_index = random.randint(0, len(parents) - 1)
        female_index = random.randint(0, len(parents) - 1)
        if male_index != female_index:
            male = parents[male_index]
            female = parents[female_index]

            # routeindex = random.sample(range(0,noOfVehicles),int(noOfVehicles/2))

            child=[]
            used=set([0])
            # for index in routeindex:
            #  child.append(male[index])
            #     for node in male[index]:
            #         used.add(node)

            for i in range(len(male)):
                comb=male[i]+female[i]
                comb= list(dict.fromkeys(comb))
                comb=  [x for x in comb if x not in used]
                set1= set(comb)
                used.update(set1)
                child.append([0]+comb+[0])

            children.append(child)

    return children

def profit_children(Solution):
    res = []
    for currOptimalSolution in Solution:
        s=[]
        for vehicle in currOptimalSolution:
            oneroute = [vehicle, checkFeasible(vehicle)]
            s.append(oneroute)
        res.append(s)
    return res

def mutation(children):
    for i in range(len(children)):
        if random.random() < mutation_rate:
            child = children[i]
            for j in range(random.randint(1, len(child) - 4)):
                child = NEIGHBOR_VRP(child,nn, data=None)
            children[i] = child
    return children


def find_max(pop):
    max=[]
    for s1 in pop:
        profit = Profit_Solution(s1)
        max.append((s1, profit))
    max.sort(key=lambda x: x[1], reverse=True)
    return max[0]



pop= get_population(s0,nn)

retain_rate=0.3
random_select_rate=0.5
mutation_rate=0.4
#
# selected = selection(pop)
# # selected=rank_route(selected)
# grandpare = remove_profit(selected)
#
# crossed = crossover(grandpare)
# children = profit_children(crossed)
# children=mutation(children)
#
# pop= selected+children
# # pop = get_population(newpop,nn)
# selected = selection(pop)



maxprofit=0
register = []
i = 0
itter_time=100

while i < itter_time:
    # 选择繁殖个体群
    selected = selection(pop)
    # selected=rank_route(selected)
    grandpare = remove_profit(selected)

    # 交叉繁殖
    crossed = crossover(grandpare)
    children= profit_children(crossed)
    # 变异操作
    children = mutation(children)
    # 更新种群
    pop = selected + children
    [bestroute, bestprofit] = find_max(pop)
    if maxprofit < bestprofit:
        maxprofit = bestprofit
        # print('New best profit is ' + str(maxprofit))
        # print('New best route is ' + str(bestroute))
    i=i+1

print('Current population is :')
print(*pop,sep='\n')
print('---------------------------------------------------------------------')
# print('The best profit is ' + str(maxprofit))
# print('The best route is ' + str(bestroute))
print(maxprofit)

# def decoding2(corssed):
#     decoded=[]
#     for routes in crossed:
#         start = routes.index(0)
#         if routes[start + 1] == 0:
#             start += 1
#
#         newroutes = routes[start:] + routes[:start]
#         routes = newroutes
#         routes.append(0)
#         # print(routes)
#         l = []
#         for w in routes:
#             if start: start = l.append([])  # l.append() returns None, that is falsey...
#             if w != 0:
#                 l[-1].append(w)
#             if w == 0: start = 1
#
#         res = []
#         for m in l:
#             if m != []:
#                 res.append([0] + m + [0])
#         while len(res) < noOfVehicles:
#             res.append([0, 0])
#
#         # soluion0 = [res, checkFeasible(res)]
#         # soluion0.append(oneroute)
#         decoded.append(res)
#     return decoded


# s2=[]
# for n_ in crossed_res:
#     oneroute_=[n_, checkFeasible(n_)]
#     s2.append(oneroute_)

# print(allNodes[25].load)
#

