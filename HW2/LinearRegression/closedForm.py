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
    closedFormSol()

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

def closedFormSol():
    X_sqr_inv = inv(np.matmul(houseData.T, houseData))
    coefficients =  np.matmul( np.matmul(X_sqr_inv,houseData.T),housePrice)
    print(coefficients)
main()