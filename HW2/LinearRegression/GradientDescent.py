import numpy as np
import math 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
dataPath = "./data.txt"
houseData = [[0]* 4 for i in range(47)]
trainData_std = [[0]* 2 for i in range(47)]
numberOfData = 47
coefficients = []
iterations = 300000
housePrice = np.ones(numberOfData)
trainData= [[0]* 2 for i in range(47)]

def main():
    read_data()
    cost,i = gradient_descent()
    plot_cost(cost,i)
def read_data():
    global houseData, trainData_std, housePrice,trainData
    with open(dataPath) as f:
        data = f.readlines()
    for row in range(numberOfData):
        houseData[row] = data[row].replace("\n","").split(',')
    for row in range(numberOfData):
        for col in range(2):
            trainData[row][col] = float(houseData[row][col])
        housePrice[row] = houseData[row][2]
    
    trainData = np.asarray(trainData)
    
    scaler = StandardScaler().fit(trainData)
    trainData_std = scaler.transform(trainData)
    trainData_std = np.insert(trainData_std, 0, values=1.0, axis=1)
    trainData = np.insert(trainData, 0, values=1.0, axis=1)
    
    # print(trainData_std)

def h_func(coefficients):
    h = np.zeros(numberOfData, dtype= float)
    for row in range(numberOfData):
        for col in range(3):
            h[row] = h[row] + (trainData_std[row][col]*coefficients[0][col])
    # print(h)
    return h
            
def gradient_descent():
    alpha = 0.01
    #initial coefficients randomly
    global coefficients
    coefficients = np.random.randn(1,3)
    # print(coefficients)
    # h = h_func(coefficients)
    # print("h = "+ str(h))
    newW = np.zeros(3)
    costD = 1.0
    cost =np.empty(iterations,dtype=float)
    i=0
    wChanges=1.0
    # coefficients_history =np.zeros((iterations,2))
    while wChanges>=0.001:
        # print(coefficients)
        h = h_func(coefficients)
        print(h)
        for col in range(3):
            costD =0.0
            for row in range(numberOfData):
                costD = costD +(( h[row] - housePrice[row])*trainData_std[row][col])
                cost[i] = cost[i] + ( h[row] - housePrice[row])*( h[row] - housePrice[row])
            costD = costD/(numberOfData)
            # print(costD)
            cost[i] = cost[i]/ (2*numberOfData)
            newW[col] = coefficients[0][col] - alpha* costD

        wChanges = math.sqrt(pow(newW[0] - coefficients[0][0],2) +  pow(newW[1] - coefficients[0][1],2) +  pow(newW[2] - coefficients[0][2],2))
        # print(wChanges)
        coefficients[0][0] = newW[0]
        coefficients[0][1] = newW[1]
        coefficients[0][2] = newW[2]
        
        i=i+1
    
    print(cost[:i])
    print(i)
    return cost,i
def plot_cost(cost,i):
    plt.figure(figsize=(12,8))
    plt.plot(range(i), cost[:i],'b.')
    plt.xlabel("Iterations")
    plt.ylabel("J(Theta)")
    plt.savefig("GD_cost")
main()




