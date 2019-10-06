import numpy as np
import math 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
dataPath = "./data.txt"

#constants
numberOfData = 47
train_size = 31
test_size = 16
iterations = 30
# data
houseData = np.zeros((numberOfData,2))
housePrice = np.ones(numberOfData)
coefficients = np.ones((1,3))


def main():
    read_data()
    cost,i, cost_train = stochastic_gradient_descent()
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
    data = np.asarray(data)
    for row in range(numberOfData):
        for col in range(2):
            houseData[row][col] = float(data[row][col])
        housePrice[row] = float(data[row][2])
    
    houseData = np.asarray(houseData)
    
    scaler = StandardScaler().fit(houseData)
    houseData = scaler.transform(houseData)
    houseData = np.insert(houseData, 0, values=1.0, axis=1)
   

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
            
def stochastic_gradient_descent():
    alpha = 0.01
    #initial coefficients randomly
    global coefficients, houseData,housePrice
    coefficients = np.random.randn(1,3)
    costD = 0.0
    cost =np.zeros(iterations,dtype=float)
    i=0
    h = np.zeros(train_size, dtype= float)

    while i<iterations:
         # shuffle data
        houseData, housePrice = shuffle(houseData, housePrice)
        h = h_func(coefficients,houseData)
        for row in range(numberOfData):
            for col in range(3):
                costD = (( h[row] - housePrice[row])*houseData[row][col])
                coefficients[0][col] = coefficients[0][col] - alpha* costD
            h = h_func(coefficients,houseData)
    
        for row in range(numberOfData):
            cost[i] =cost[i]+ pow( h[row] - housePrice[row],2)
        cost[i] = cost[i]/ (2*numberOfData)
       
        i=i+1
    
    print(cost[:i])
    cost_train = getCost(h,housePrice)
    return cost,i, cost_train

# def linearRegression(coefficients,data):
#     predict_price = h_func(coefficients, data)
#     cost_test = getCost(predict_price,price_test)
#     print("cost test: " + str(cost_test))
#     return cost_test

    




def plot_cost(cost,i):
    plt.figure(figsize=(12,8))
    x_ticks = np.arange(1, i, 1)
    plt.xticks(x_ticks)
    plt.plot(range(i), cost[:i],'b.')
    plt.xlabel("Iterations")
    plt.ylabel("J(Theta)")
    plt.savefig("SGD_cost")
main()




