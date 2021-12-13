#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Michael Lust 
#ECGR-4115 
#Convex Final Project
#December 14, 2021

# Imports
import math
import random
import pandas as pd
from heapq import heapify, heappush, heappop # This is for the min heap
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np # probably not necessary, but is used for initializing the edge and weights
import sys # used for setting the costs to inf (could be replaced with just using a really large number)
import time # used to get the runtime of the algorithm
from matplotlib import pyplot as plt # used to graph the path taken
import cvxpy as cp #CVX program


# In[2]:


# Set up variables for DataCenter
numNodes = 18
edgeMatrix = np.zeros(shape = [numNodes, numNodes], dtype = int)
weightMatrix = np.zeros(shape = [numNodes, numNodes], dtype = float)
nodes = []
nodeCounter = 0
nodeCost = []
explored = [False] * numNodes
openList = []


# In[3]:


#Mapping Node to reallocate the workload to other nodes
mappingNode = {
    "MappingNode1": [18, 12],
    "MappingNode2": [10, 12],
    "MappingNode3": [1, 12]
}


# In[4]:


# defines coordinates of each data center as a big node (K=4)
dataCenter = {
    "DataCenter1": [18, 25],
    "DataCenter2": [10, 25],
    "DataCenter3": [1, 25], 
}


# In[5]:


smallNode = {
    "Node1": [25, 25],
    "Node2": [23, 25],
    "Node3": [21, 25],
    "Node4": [19, 25],
    "Node5": [17, 25],
    "Node6": [15, 25],
    "Node7": [13, 25],
    "Node8": [11, 25],
    "Node9": [9, 25],
    "Node10": [7, 25],
    "Node11": [5, 25],
    "Node12": [2, 25]
}


# In[6]:


# Node class
class Node(): # defining nodes
    nodeID = -1
    xCoord = 0
    yCoord = 0
    weight = 0
    prevNode = None
    isDataCenter = False
    isDataCenter1 = False 
    isDataCenter2 = False
    isDataCenter3 = False
    isMappingNode = False
    isCost = False
    
    # instantiate a node with given coordinates
    def __init__ (self, xCoord, yCoord, weight):
        global nodeCounter
        self.xCoord = xCoord
        self.yCoord = yCoord
        self.weight = weight
        self.nodeID = nodeCounter
        nodeCounter += 1
    
    # used for less than comparisons (min heap)
    def __lt__(self, other):
        global nodeCost
        if nodeCost[self.nodeID] < nodeCost[other.nodeID]:
            return True
        else:
            return False
        
    # keep track of what node you came from
    def setPrevNode(self, node):
        self.prevNode = node
        
    # set as MappingNode to be used and plotted.
    def setMappingNode(self):
        self.isMappingNode = True
    
    # set as DataCenter to be plotted.
    def setDataCenter(self):
        self.isDataCenter = True
    
    # set as DataCenter to be used.
    def setDataCenter1(self):
        self.isDataCenter1 = True

    # set as DataCenter to be used.
    def setDataCenter2(self):
        self.isDataCenter2 = True
        
    # set as DataCenter to be used.
    def setDataCenter3(self):
        self.isDataCenter3 = True
        
    def setCost(self):
        self.isCost = True
        
    # set as smaller nodes to be use
    #def setSmallNode(self):
        #self.isSmallNode = True


# In[7]:


# defining the dataCenter workload from each mappingNode
#This can be thought as Cost from external source being fed into mappingNode
def workLoad():
    weight_cost = np.random.randint(low = 0 , high = 500, size=(12,)) 
    return weight_cost


# In[8]:


#Testing a different workLoad function to best be compatable with CVXPY. The new version multiplies matrixes differently.
def workLoad2():
    power = np.random.randint(low = 0 , high = 5, size=(3,)) 
    bandwidth = np.random.randint(low = 0, high = 50, size = (3,3))
    x = np.random.randint(low = 0, high = 50, size = (3,3))
    y =  np.random.randint(low = 0 , high = 5, size=(3,)) 

    power = sum(power)
    bandwidth = sum(sum(bandwidth))
    left = power * (pow((y),2))
    right = bandwidth * (pow((y),2))
    work_load = left + right
    return work_load


# In[9]:


test = workLoad2()
test


# In[10]:


# defining the dataCenter cost from each node
def minCost(X,Y): #Cost function to minimize
    power = np.random.randint(low = 0 , high = 5, size=(3,)) 
    bandwidth = np.random.randint(low = 0, high = 50, size = (3,3))

    for i in range(2):
        for j in range(2):
            power[i] = power[i] * (pow((Y[i]),2))
            bandwidth[i][j] = bandwidth[i][j] * (pow((X[i][j]),2))
    bandwidth = sum(bandwidth)
    total = power + bandwidth #Minimize total
    total = sum(total)
    return total


# In[11]:


#Testing minCost function
x = np.random.randint(low = 0 , high = 5, size=(3,3)) 
y = np.random.randint(low = 0, high = 50, size = (3))
test = minCost(x,y)
test


# In[12]:


def powerCost(Y):
    for i in range(2):
        power[i] = power[i] * (pow((Y[i]),2))
    return power


# In[13]:


def bandwidthCost(X):
    for i in range(2):
        for j in range(2):
            bandwidth[i][j] = bandwidth[i][j] * (pow((X[i][j]),2))
    return bandwidth


# In[14]:


nodeCost = workLoad()
nodeCost


# In[15]:


zeroMatrix = np.zeros(6)


# In[16]:


nodeCost = np.concatenate((nodeCost,zeroMatrix), axis = None)
nodeCost


# In[17]:


#Setting up the parameters to be added to the node class
node1 = (Node(smallNode['Node1'][0],smallNode['Node1'][1],nodeCost[0]))
node1.setDataCenter1()

node2 = (Node(smallNode['Node2'][0],smallNode['Node2'][1],nodeCost[1]))
node2.setDataCenter1()

node3 = (Node(smallNode['Node3'][0],smallNode['Node3'][1],nodeCost[2]))
node3.setDataCenter1()

node4 = (Node(smallNode['Node4'][0],smallNode['Node4'][1],nodeCost[3]))
node4.setDataCenter1()

node5 = (Node(smallNode['Node5'][0],smallNode['Node5'][1],nodeCost[4]))
node5.setDataCenter2()

node6 = (Node(smallNode['Node6'][0],smallNode['Node6'][1],nodeCost[5]))
node6.setDataCenter2()

node7 = (Node(smallNode['Node7'][0],smallNode['Node7'][1],nodeCost[6]))
node7.setDataCenter2()

node8 = (Node(smallNode['Node8'][0],smallNode['Node8'][1],nodeCost[7]))
node8.setDataCenter2()

node9 = (Node(smallNode['Node9'][0],smallNode['Node9'][1],nodeCost[8]))
node9.setDataCenter3()

node10 = (Node(smallNode['Node10'][0],smallNode['Node10'][1],nodeCost[9]))
node10.setDataCenter3()

node11 = (Node(smallNode['Node11'][0],smallNode['Node11'][1],nodeCost[10]))
node11.setDataCenter3()

node12 = (Node(smallNode['Node12'][0],smallNode['Node12'][1],nodeCost[11]))
node12.setDataCenter3()

mappingNode1 = (Node(mappingNode['MappingNode1'][0],mappingNode['MappingNode1'][1],nodeCost[12]))
mappingNode1.setMappingNode()

mappingNode2 = (Node(mappingNode['MappingNode2'][0],mappingNode['MappingNode2'][1],nodeCost[13]))
mappingNode2.setMappingNode()

mappingNode3 = (Node(mappingNode['MappingNode3'][0],mappingNode['MappingNode3'][1],nodeCost[14]))
mappingNode3.setMappingNode()

dataCenter1 = (Node(dataCenter['DataCenter1'][0],dataCenter['DataCenter1'][1],nodeCost[15]))
dataCenter1.setDataCenter()

dataCenter2 = (Node(dataCenter['DataCenter2'][0],dataCenter['DataCenter2'][1],nodeCost[16]))
dataCenter2.setDataCenter()

dataCenter3 = (Node(dataCenter['DataCenter3'][0],dataCenter['DataCenter3'][1],nodeCost[17]))
dataCenter3.setDataCenter()


# In[18]:


#Adding the parameters of xCoord, yCoord, and weight to the node class
nodes.append(node1)
nodes.append(node2)
nodes.append(node3)
nodes.append(node4)
nodes.append(node5)
nodes.append(node6)
nodes.append(node7)
nodes.append(node8)
nodes.append(node9)
nodes.append(node10)
nodes.append(node11)
nodes.append(node12)
nodes.append(mappingNode1)
nodes.append(mappingNode2)
nodes.append(mappingNode3)
nodes.append(dataCenter1)
nodes.append(dataCenter2)
nodes.append(dataCenter3)
nodes


# In[19]:


#Appending the total weights from each node in each dataCenter.
cost_DC1 = 0
cost_DC2 = 0
cost_DC3 = 0

for i in range (numNodes):
    if nodes[i].isDataCenter1:
        cost_DC1 += nodes[i].weight

for i in range (numNodes):
    if nodes[i].isDataCenter2:
        cost_DC2 += nodes[i].weight

for i in range (numNodes):
    if nodes[i].isDataCenter3:
        cost_DC3 += nodes[i].weight

print(cost_DC1)
print(cost_DC2)
print(cost_DC3)


# In[20]:


#Updating the cost C of each dataCenter when allocating workload.
def newCost():
    updatedCost = workLoad()
    newDataCenter1 = updatedCost[0] + updatedCost[1] + updatedCost[2] + updatedCost[3]
    newDataCenter2 = updatedCost[4] + updatedCost[5] + updatedCost[6] + updatedCost[7]
    newDataCenter3 = updatedCost[8] + updatedCost[9] + updatedCost[10] + updatedCost[11]
    updatedCost = [newDataCenter1, newDataCenter2, newDataCenter3]
    return updatedCost


# In[21]:


#Showing a simple graph of the nodes being able to be plotted
for i in range(numNodes):
    if nodes[i].isDataCenter or nodes[i].isMappingNode:
        plt.plot(nodes[i].yCoord, nodes[i].xCoord, marker = "o") 


# In[22]:


node_direction = [1,-1]
A = random.choice(node_direction)
A


# In[23]:


b = ([nodes[15].weight,nodes[16].weight,nodes[17].weight])
b 


# In[24]:


Cost_T0 = ([cost_DC1, cost_DC2, cost_DC3])
Cost_T0


# In[25]:


#Testing convering minimization problem to scalar
def costFunction(X,Y):
    power = np.random.randint(low = 0 , high = 5, size=(3,)) 
    bandwidth = np.random.randint(low = 0, high = 50, size = (3,3))
    
    power = sum(power)
    bandwidth = sum(sum(bandwidth))
    pow_sum = sum(power * (pow((Y),2)))
    band_sum = sum(bandwidth * (pow((X),2)))
    all_sum = pow_sum + band_sum
    total = sum(all_sum)
    return total


# In[26]:


#Testig costFunction to see what values are likely
x = np.random.randint(low = 0 , high = 5, size=(3,3)) 
y = np.random.randint(low = 0, high = 50, size = (3))
totally = costFunction(x,y)
totally


# In[27]:


#Algorithm to allocate resources given cost function
def resourceAllocate_1(dataCenterCost):
    #Use of Linear Programming to minimize the cost for resource allocation
    #Using CVX in Python
    
    #node_direction = [-1,1]
    node_direction = [-1]
    #We are keeping the workload as always sending to not back up the network; nodes on receiving 
    #workload continuously at random will increase conjestion and optimization will be impossible
    workload_treated = [-1,1] #Workload is shown as treated always so it is not used
    A = random.choice(node_direction) #Connection entering or leaving node
    b = np.zeros(len(dataCenterCost))
    
    for i in range (len(dataCenterCost)):
        b[i] = nodes[i + 12].weight + dataCenterCost[i] #exogenous workload to be reallocated
    
    # Construct the problem.
    x = cp.Variable((len(b),len(b)), PSD=True) # Creates a 3 by 3 positive semidefinite variable.
    y = cp.Variable(len(b))
    
    #Now doing the CVX minimization
    #objective = cp.Minimize(sum(power * (pow((y),2))) + sum(sum(bandwidth * (pow((x),2)))))
    #objective = cp.Minimize(sum(power * (pow((y),2)) + bandwidth * (pow((x),2))))
    objective = cp.Minimize(costFunction(x,y))
    #constraints = [0 <= x, x <= 1]
    constraints = [x[0][0] + x[1][0] + x[2][0] == y[0],
                   x[0][1] + x[1][1] + x[2][1] == y[1],
                   x[0][0] + x[1][0] + x[2][0] == y[2],
                   
                   b[0] == x[0][0] + x[0][1] + x[0][2],
                   b[1] == x[1][0] + x[1][1] + x[1][2],
                   b[2] == x[2][0] + x[2][1] + x[2][2]]
    
    prob = cp.Problem(objective, constraints)
    prob.solve()

    # Print result.
    #print("\nThe optimal value is", prob.value) #CVX is having issues solving this. It is just the sum of the 
                                                 # Y matrix. I tried a full day to solve but to no avail.
    print("This is for problem 1")
    print("A solution x is")
    print(x.value) 
    print("A solution y is")
    print(y.value)
    return prob.value, x.value, y.value


# In[28]:


#Algorithm to allocate resources given cost function with updated parameters
def resourceAllocate_2(dataCenterCost):
    #Use of Linear Programming to minimize the cost for resource allocation
    #Using CVX in Python
    
    node_direction = [-1,1]
    workload_treated = [-1]
    A = random.choice(node_direction) #Connection entering or leaving node
    Y_direction = random.choice(workload_treated) 
    b = np.zeros(len(dataCenterCost))
    x_max = np.zeros(len(dataCenterCost))
    y_untreated = np.zeros(len(dataCenterCost)) #This is a y matrix that represents no workload has been treated.
    
    for i in range (len(dataCenterCost)):
        b[i] = nodes[i + 12].weight + dataCenterCost[i] #exogenous workload to be reallocated
    
    # Construct the problem.
    x = cp.Variable((len(b),len(b)), PSD=True) # Creates a 3 by 3 positive semidefinite variable.
    y = cp.Variable(len(b))
    #objective = cp.Minimize(cp.sum(calculate_scalar(K_Data_Center_Predict, Mapping_Nodes_J, theta)))
    
    #Now doing CVX minimization
    #objective = cp.Minimize(sum(power * (pow((y),2))) + sum(sum(bandwidth * (pow((x),2)))))
    #objective = cp.Minimize(sum(power * (pow((y),2)) + bandwidth * (pow((x),2))))
    objective = cp.Minimize(costFunction(x,y))
    
    for i in range((len(b))):
        if x_max[i] < b[i]:
            x_max[i] = b[i] 
            
    constraints = [x[0][0] + x[1][0] + x[2][0] == y[0],
                   x[0][1] + x[1][1] + x[2][1] == y[1],
                   x[0][0] + x[1][0] + x[2][0] == y[2],
                   
                   b[0] == x[0][0] + x[0][1] + x[0][2],
                   b[1] == x[1][0] + x[1][1] + x[1][2],
                   b[2] == x[2][0] + x[2][1] + x[2][2]]
    
    
    prob = cp.Problem(objective, constraints)
    
    if A < 0: #If the link is sending
        prob.solve()

        # Print result.
        #print("\nThe optimal value is", prob.value)
        print("This is for problem 2")
        print("A solution x is")
        print(x.value)
        print("A solution y is")
        print(y.value)
        
        for i in range(len(b)):
            for j in range(len(b)):
                nodes[i + 12].weight = b[i] + nodes[i + 12].weight #b[i] queues workload into mapping nodes, adds leftovers
                if x[i][j].value == None:
                    x.value = np.zeros((3,3))
                b[i] = b[i] - x[i][j].value #Workload is scheduled into x[i] to take away from mapping nodes, b[i] is leftovers
                if b[i] <= 0: #If the workload taken off is more than in the queue, the queue will be set to zero instead of
                    b[i] = 0  #going into the negatives.
                nodes[i + 15].weight = x[i][j].value + nodes[i + 15].weight #dataCenter Nodes receives weights distributed by CVX optimization
                nodes[i + 12].weight = b[i] #Mapping node is updated by any left over b[i]
                if Y_direction <= 0: #If y < 0: The workload is counted as treated and weights reset. If not, they are left queued.
                    nodes[i + 15].weight = 0 
                    nodes[i + 12].weight = 0
            
    elif A > 0: #If the link is receiving 
        node_weight = []
        added_weight = 0
        for i in range(len(b)): 
            nodes[i + 12].weight = nodes[i + 12].weight + b[i] #Workload scheduled is added to the queue for mappingData
            added_weight = nodes[12].weight + nodes[13].weight + nodes[14].weight
            node_weight.append(nodes[12].weight)
            node_weight.append(nodes[13].weight)
            node_weight.append(nodes[14].weight)
            return added_weight, node_weight, y_untreated #Returns the untreated workload split since it is ingoing.
    return prob.value, x.value, y.value #If link was sending, then return optimization values


# In[29]:


def calculate_scalar(X, Y, theta, M): #Declaring values and computing the Scalar value J
#loss is calculated by taking the mean of squared differences between actual(target) and predicted values. 
    predictions = X.dot(theta)  #Dot product of array X and theta
    errors = np.subtract(predictions,Y) #Matrix subtraction with predictions and Y
    squaringErrors = np.square(errors) #Now errors contained in matrix. We square all values in matrix error.
    J = 1/(2*M)*np.sum(squaringErrors) #Scalar equation using matrix squErrors
    return J


# In[30]:


def gradient_descent(X, Y, theta, alpha, iterations, M):  #Function to calculate gradient descent for linear regression
    
    result = np.zeros(iterations)   #creating a row of an array with an undetermined amount of zeroes.
    theta_interval = np.zeros([iterations, theta.size])  #creating an array for each interval to be plotted (X1, X2, X3) 
    
    for i in range(iterations):    #For loop with iterations as an input.
        predictions = X.dot(theta)   #Dot product of array X and theta resulting in scalar
        errors = np.subtract(predictions,Y) #Matrix subtration between predictions and value Y
        sum_delta = (alpha/M)*X.transpose().dot(errors); #learning rate over training examples * scalar of resulting dot product.  
        theta = theta-sum_delta;   #Current theta minus scalar sum_delta for final value of theta                      
        result[i] = calculate_scalar(X, Y, theta, M)
        theta_interval[i] = theta #Needed to show the previous thetas used for the resulting scalar.

    return theta, result, theta_interval


# In[31]:


#Running the algorithm with input of time_slot
time_slot = 100

X_1 = np.zeros(time_slot)
X_val_1 = np.zeros((time_slot,3,3))

X_2 = np.zeros(time_slot)
X_val_2 = np.zeros((time_slot,3,3))

Y_1 = np.zeros((time_slot,3))
Y_2 = np.zeros((time_slot,3))

for i in range(time_slot):
    Cost_T_1 = newCost()
    X_1[i], X_val_1[i], Y_1[i] = resourceAllocate_1(Cost_T_1)
    i += 1
    
for j in range(time_slot):
    Cost_T_2 = newCost()
    X_2[j], X_val_2[j], Y_2[j] = resourceAllocate_2(Cost_T_2)
    j += 1


# In[32]:


print(len(X_1))
print(len(X_2))


# In[33]:


print(X_val_1)
print(X_val_2)


# In[34]:


#Given Prob.value is not working properly, the X_1 and X_2 values are added manually
for i in range(len(X_1)):
    X_1[i] = sum(Y_1[i])
    
for i in range(len(X_2)):
    X_2[i] = sum(Y_2[i])
X_1


# In[35]:


#Plotting the best fit line for Network Conjestion from Resource Allocation 1
ones_fit_1 = X_1.reshape(len(X_1),1)
X_fit_1 = np.append(ones_fit_1, np.ones((len(X_1),1)), axis = 1)
time_slot_fit_1 = np.arange(time_slot).reshape((time_slot, 1))

# Calculating the parameters using the least square method
theta_1 = np.linalg.inv(X_fit_1.T.dot(X_fit_1)).dot(X_fit_1.T).dot(time_slot_fit_1)

print(f'The parameters of the line: {theta_1}')

# Now, calculating the y-axis values against x-values according to
# the parameters theta0 and theta1

y_line_1 = X_fit_1.dot(theta_1)

plt.scatter(time_slot_fit_1, ones_fit_1, color = 'Blue', label = 'Cost' )
#plt.plot(y_line_1, X_fit_1, color = 'Red', label = 'BestFitLine') Best fit line was not showing properly
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel('Time')
plt.ylabel('Value of Cost at different time slots')
plt.title('Workloads treated at different time slots for Network Congestion Optimization')
#plt.xlim([1500, 5000]) #Gets rid of any outliers for second BestFitLine
#plt.ylim([10000,35000])
plt.legend()
plt.show()


# In[36]:


#Plotting the best fit line for Network Conjestion from Resource Allocation 1
ones_fit_2 = X_2.reshape(len(X_2),1)
X_fit_2 = np.append(ones_fit_2, np.ones((len(X_2),1)), axis = 1)
time_slot_fit_2 = np.arange(time_slot).reshape((time_slot, 1))

# Calculating the parameters using the least square method
theta_2 = np.linalg.inv(X_fit_2.T.dot(X_fit_2)).dot(X_fit_2.T).dot(time_slot_fit_2)

print(f'The parameters of the line: {theta_2}')

# Now, calculating the y-axis values against x-values according to
# the parameters theta0 and theta1

y_line_2 = X_fit_2.dot(theta_2)

plt.scatter(time_slot_fit_2, ones_fit_2, color = 'Blue', label = 'Cost' )
#plt.plot(y_line_2, X_fit_2, color = 'Red', label = 'BestFitLine') Best fit line was not showing properly
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel('Time')
plt.ylabel('Value of Cost at different time slots')
plt.title('Workloads treated at different time slots for Network Congestion Optimization')
#plt.xlim([1, 8500]) #Gets rid of any outliers for second BestFitLine
#plt.ylim([1, 100000])
plt.legend()
plt.show()


# In[37]:


#Testing Gradient Descent for both Network ALlocation algorithms


# In[38]:


#Allocating new arrays to represent optimization values to be tested.
X_val_1_DC1 = np.zeros((time_slot,3))
X_val_1_DC2 = np.zeros((time_slot,3))
X_val_1_DC3 = np.zeros((time_slot,3))

X_val_2_DC1 = np.zeros((time_slot,3))
X_val_2_DC2 = np.zeros((time_slot,3))
X_val_2_DC3 = np.zeros((time_slot,3))

for i in range(len(X_val_1)):
    for j in range(2):
        X_val_1_DC1[i][0] = X_val_1[i][j][0]
        X_val_1_DC2[i][1] = X_val_1[i][j][1]
        X_val_1_DC3[i][2] = X_val_1[i][j][2]  

for i in range(len(X_val_2)):
    for j in range(2):
        X_val_2_DC1[i][0] = X_val_2[i][j][0]
        X_val_2_DC2[i][1] = X_val_2[i][j][1]
        X_val_2_DC3[i][2] = X_val_2[i][j][2]


# In[39]:


X_val_1_DC2


# In[40]:


np.delete(X_val_1_DC2,[0,2],1)


# In[41]:


#Total workload that has been treated to be compared to the optimal solution value.
Y_1_row = np.array(Y_1)
Y_1 = Y_1_row.sum(axis = 1)
Y_1


# In[42]:


#Total workload that has been treated to be compared to the optimal solution value.
Y_2_row = np.array(Y_2)
Y_2 = Y_2_row.sum(axis = 1)
Y_2


# In[43]:


#Reshape is performed to run standard scaling so that gradient descent can be run.
M = time_slot
Y_T_1 = Y_1
Y_T_1 = Y_T_1.reshape(M,1)
Y_T_1


# In[45]:


#Reshape is performed to run standard scaling so that gradient descent can be run.
Y_T_2 = Y_2
Y_T_2 = Y_T_2.reshape(M,1)
Y_T_2[np.isnan(Y_T_2)] = 0
for i in range(len(Y_2)):
    if Y_T_2[i][0] <= 0:
        Y_T_2[i] = Y_T_2[i-1][0]
Y_T_2


# In[46]:


#Structuring the dataset to test linear regression
Ones_T = np.ones((M,1))

X_val_1_DC1 = np.delete(X_val_1_DC1,[1,2],1)
X_val_1_DC2 = np.delete(X_val_1_DC2,[0,2],1)
X_val_1_DC3 = np.delete(X_val_1_DC3,[0,1],1)


X_val_2_DC1 = np.delete(X_val_2_DC1,[1,2],1)
X_val_2_DC2 = np.delete(X_val_2_DC2,[0,2],1)
X_val_2_DC3 = np.delete(X_val_2_DC3,[0,1],1)


X_T_1 = np.hstack((Ones_T, X_val_1_DC1, X_val_1_DC2, X_val_1_DC3))
print(X_T_1)

X_T_2 = np.hstack((Ones_T, X_val_2_DC1, X_val_2_DC2, X_val_2_DC3))
print(X_T_2)


Y_T_1 = np.hstack((Ones_T,Y_T_1))
Y_T_2 = np.hstack((Ones_T,Y_T_2))


# In[47]:


#Setting the data into standardized scale so that gradient descent model can be trained.
scalar = StandardScaler()

X_T_1 = scalar.fit_transform(X_T_1)
Y_T_1 = scalar.fit_transform(Y_T_1)
Y_Ts_1 = np.zeros(M)

X_T_2 = scalar.fit_transform(X_T_2)
Y_T_2 = scalar.fit_transform(Y_T_2)
Y_Ts_2 = np.zeros(M)

for i in range(time_slot):
    Y_Ts_1[i] = (Y_T_1[i][1])

for i in range(time_slot):
    Y_Ts_2[i] = (Y_T_2[i][1])
Y_Ts_2


# In[48]:


#We are comparing the output of y and the workload of x to see if they match. At different time slots the optimal value
#does not match the y output so we go ahead compare the two to see how well the model is performing.
#Making a theta array with initializations of O and setting validation parameters.
theta_1 = np.zeros(4)
theta_2 = np.zeros(4)
iterations = time_slot
alpha = 0.1 #This is to avoid getting overfill error.

result_1 = calculate_scalar(X_T_1, Y_Ts_1, theta_1, M)
print('Scalar value for result 1 is ', result_1 ) #Print the scalar value for Gradient Descent

result_2 = calculate_scalar(X_T_2, Y_Ts_2, theta_2, M)
print('Scalar value for result 2 is ', result_2 ) #Print the scalar value for Gradient Descent


#Calculating gradient descent with theta and scalar J for validation set
theta, result_1, theta_interval = gradient_descent(X_T_1, Y_Ts_1, theta_1, alpha, iterations, M)
print('Final value of theta_1 =', theta_1)
print('Y_1 = ', result_1)

#Calculating gradient descent with theta and scalar J for validation set
theta, result_2, theta_interval = gradient_descent(X_T_2, Y_Ts_2, theta_2, alpha, iterations, M)
print('Final value of theta_2 =', theta_2)
print('Y_2 = ', result_2)


# In[49]:


#Plotting the Scalar J vs. Number of Iterations for all X values combined
plt.plot(range(1, iterations + 1), result_1, color='Red', label = 'ResAlo1' )
plt.plot(range(1, iterations + 1), result_2, color='Blue', label = 'ResAlo2' )
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel('Time')
plt.ylabel('Value of Loss at different time slots')
plt.title('Convergence for Gradient Descent for Network Congestion Optimization')
plt.legend()
plt.show()


# In[ ]:




