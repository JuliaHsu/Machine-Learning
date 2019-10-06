from numpy.linalg import inv
import numpy as np
import math 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
dataPath = "./data.txt"
numberOfData = 47
iterations = 3000
houseData = np.zeros((numberOfData,2))
housePrice = np.ones(numberOfData)
coefficients = np.ones((1,3))

def main():
    read_data()
    cost_train = closedFormSol()
    Erms_train = getErms(cost_train,numberOfData)
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
            h[row] = h[row] + (data[row][col]*coefficients[col])
    # print(h)
    return h

def getCost(h_price, true_price):
    cost =0.0
    for row in range(h_price.shape[0]):
        cost = cost+ pow((h_price[row]- true_price[row]),2)
    
    cost = cost/(2*h_price.shape[0])
    return cost

def getErms(cost,data_size):
    Erms = math.sqrt(2*cost/(data_size))
    return Erms

def closedFormSol():
    global coefficients
    X_sqr_inv = inv(np.matmul(houseData.T, houseData))
    coefficients =  np.matmul( np.matmul(X_sqr_inv,houseData.T),housePrice)
    predict = h_func(coefficients, houseData)
    cost_train = getCost(predict, housePrice)
    return cost_train

main()