import numpy as np
import math 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
dataPath = "./data.txt"
numberOfData = 47
iterations = 5000
houseData = [[0]* 3 for i in range(numberOfData)]
housePrice = np.ones(numberOfData)
coefficients = np.ones((1,3))
reg = 0.05

def main():
    read_data()
    cost,i,cost_train = gradient_descent_reg()
    Erms_train = getErms(cost_train, numberOfData)
    plot_cost(cost,i)
    print("number of iterations: "+ str(i))
    print("coefficients: "+ str(coefficients))
    print("cost train: "+ str(cost_train))
    print("Erms train: "+ str(Erms_train))
def read_data():
    global houseData, housePrice
    with open(dataPath) as f:
        data = f.readlines()
    for row in range(numberOfData):
        data[row] = data[row].replace("\n","").split(',')
    for row in range(numberOfData):
        for col in range(2):
            houseData[row][col] = float(data[row][col])
        housePrice[row] = float(data[row][2])
    
    houseData = np.asarray(houseData)
    
    scaler = StandardScaler().fit(houseData)
    houseData = scaler.transform(houseData)
    houseData = np.insert(houseData, 0, values=1.0, axis=1)
    


def h_func(coefficients,data):
    h = np.zeros(data.shape[0], dtype= float)
    for row in range(data.shape[0]):
        for col in range(3):
            h[row] = h[row] + (data[row][col]*coefficients[0][col])
    # print(h)
    return h

def getCost(h_price, true_price):
    cost =0.0
    sumTheta = 0.0
    for row in range(h_price.shape[0]):
        cost = cost+ pow((h_price[row]- true_price[row]),2) 

    #add lambda for regularization, reg = 0.1
    for col in range(3):
        sumTheta = sumTheta+ pow(coefficients[0][col],2)
    cost = (cost + (reg * sumTheta))/(2*h_price.shape[0])
    # print(cost)
    return cost

def getErms(cost,data_size):
    Erms = math.sqrt(2*cost/(data_size))
    return Erms
            
def gradient_descent_reg():
    alpha = 0.01
    #initial coefficients randomly
    global coefficients
    coefficients = np.random.randn(1,3)
    newCoeff = np.zeros(3)
    costD = 1.0
    cost =np.zeros(iterations,dtype=float)
    i=0
    wChanges=1.0
    # coefficients_history =np.zeros((iterations,2))
    while wChanges>=0.001:
        # print(coefficients)
        h = h_func(coefficients,houseData)
        # print(h)
        for col in range(3):
            costD =0.0
            for row in range(numberOfData):
                costD = costD +(( h[row] - housePrice[row])*houseData[row][col])
            costD = costD/(numberOfData)
            if col ==0:
                newCoeff[col] = coefficients[0][col] - alpha* costD
            else:
                newCoeff[col] = (coefficients[0][col] - alpha* costD) - ((reg/numberOfData) * coefficients[0][col])
        
        wChanges = math.sqrt(pow(newCoeff[0] - coefficients[0][0],2) +  pow(newCoeff[1] - coefficients[0][1],2) +  pow(newCoeff[2] - coefficients[0][2],2))
        

        # print(wChanges)
        coefficients[0][0] = newCoeff[0]
        coefficients[0][1] = newCoeff[1]
        coefficients[0][2] = newCoeff[2]
        cost[i] = getCost(h,housePrice)
      
        
        
        i=i+1
    
    print(cost[:i])
    cost_train = getCost(h,housePrice)
    return cost,i, cost_train

def plot_cost(cost,i):
    plt.figure(figsize=(12,8))
    plt.plot(range(i), cost[:i],'b.')
    plt.xlabel("Iterations")
    plt.ylabel("J(Theta)")
    plt.savefig("GD_reg_cost")
main()




