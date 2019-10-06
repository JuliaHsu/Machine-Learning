import numpy as np
import math 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
dataPath = "./data.txt"

#constants
numberOfData = 47
train_size = 31
test_size = 16
iterations = 3000
# data
houseData = [[0]* 2 for i in range(numberOfData)]
housePrice = np.ones(numberOfData)
coefficients = np.ones((1,3))
trainData= [[0]* 2 for i in range(train_size)]
testData= [[0]* 2 for i in range(test_size)]
price_train = np.ones(train_size)
price_test = np.ones(test_size)

def main():
    read_data()
    cost_i, i, cost_train = gradient_descent()
    plot_cost(cost_i,i)
    cost_test = linearRegression(coefficients,testData)
    Erms_train = getErms(cost_train, train_size)
    Erms_test = getErms(cost_test, test_size)
    print("number of iterations: "+ str(i))
    print("coefficients: "+ str(coefficients))
    print("cost train: "+ str(cost_train))
    print("cost test: " + str(cost_test))
    print("Erms train: "+ str(Erms_train))
    print("Erms test: "+ str(Erms_test))
    
def read_data():
    global houseData, housePrice,trainData,testData,price_train, price_test
    with open(dataPath) as f:
        data = f.readlines()
    for row in range(numberOfData):
        data[row] = data[row].replace("\n","").split(',')

    for row in range(numberOfData):
        for col in range(2):
            houseData[row][col] = float(data[row][col])
        housePrice[row] = float(data[row][2])
    
    trainData, testData, price_train, price_test = train_test_split(houseData,housePrice,test_size=0.33,random_state = 42)
    
    trainData = np.asarray(trainData)
    scaler = StandardScaler().fit(trainData)
    trainData = scaler.transform(trainData)
    testData = scaler.transform(testData)
    trainData = np.insert(trainData, 0, values=1.0, axis=1)
    testData = np.insert(testData, 0, values=1.0, axis=1)
   

def h_func(coefficients, data):
    h = np.zeros(data.shape[0], dtype= float)
    for row in range(data.shape[0]):
        for col in range(3):
            h[row] = h[row] + (data[row][col]*coefficients[0][col])
    return h

def getCost(h_price, true_price):
    cost =0.0
    for row in range(h_price.shape[0]):
        cost = cost+ pow((h_price[row]- true_price[row]),2)
    
    cost = cost/(2*h_price.shape[0])
    # print(cost)
    return cost

def getErms(cost,data_size):
    Erms = math.sqrt(2*cost/(data_size))
    return Erms
    
            
def gradient_descent():
    alpha = 0.01
    #initial coefficients randomly
    global coefficients
    coefficients = np.random.randn(1,3)
    newCoeff = np.zeros(3)
    costD = 1.0
    cost =np.zeros(iterations,dtype=float)
    i=0
    wChanges=1.0
    h = np.zeros(train_size, dtype= float)
    # coefficients_history =np.zeros((iterations,2))
    while wChanges>=0.001:
        # print(coefficients)
        h = h_func(coefficients,trainData)
        # print(h)
        for col in range(3):
            costD =0.0
            for row in range(train_size):
                costD = costD +(( h[row] - price_train[row])*trainData[row][col])

            costD = costD/(train_size)
            # print(costD)
            newCoeff[col] = coefficients[0][col] - alpha* costD

        wChanges = math.sqrt(pow(newCoeff[0] - coefficients[0][0],2) +  pow(newCoeff[1] - coefficients[0][1],2) +  pow(newCoeff[2] - coefficients[0][2],2))
        # print(wChanges)
        coefficients[0][0] = newCoeff[0]
        coefficients[0][1] = newCoeff[1]
        coefficients[0][2] = newCoeff[2]
        cost[i] = getCost(h,price_train)
        i=i+1
    
    # print(cost[:i])
    
    
    cost_train = getCost(h,price_train)
    

    return cost,i, cost_train

def linearRegression(coefficients,data):
    predict_price = h_func(coefficients, data)
    cost_test = getCost(predict_price,price_test)
    return cost_test


def plot_cost(cost,i):
    plt.figure(figsize=(12,8))
    plt.plot(range(i), cost[:i],'b.')
    plt.xlabel("Iterations")
    plt.ylabel("J(Theta)")
    plt.savefig("GD_cv_cost")
main()




