import numpy as np
import math 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
dataPath = "./data.txt"
houseData = [[0]* 4 for i in range(47)]
houseData_std = [[0]* 4 for i in range(47)]
numberOfData = 47
coefficients = []
iterations = 1000

def main():
    read_data()
    cost = gradient_descent()
    plot_cost(cost)
def read_data():
    global houseData, houseData_std 
    with open(dataPath) as f:
        data = f.readlines()
    for row in range(numberOfData):
        houseData[row] = data[row].replace("\n","").split(',')
    for row in range(numberOfData):
        for col in range(3):
            houseData[row][col] =import numpy as np
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




 float(houseData[row][col])
    np.asarray(houseData)
    houseData = np.insert(houseData, 0, values=1, axis=1)
    # split data
    house_train, train_price, house_test, test_price = train_test_split(houseData[:,:3],houseData[:,3],test_size=0.33, random_state=42)
    scaler = MinMaxScaler()
    scaler.fit(house_train)
    houseData_std = scaler.transform(houseData)
    

def h_func(coefficients):
    h = np.zeros(numberOfData, dtype= float)
    for row in range(numberOfData):
        for col in range(3):
            h[row] = h[row] + (houseData_std[row][col]*coefficients[col])
    # print(h)
    return h
            
def gradient_descent():
    alpha = 0.1
    #initial coefficients randomly
    global coefficients
    coefficients = np.random.random_sample(3)
    # print(coefficients)
    h = h_func(coefficients)
    # print("h = "+ str(h))
    newW = np.zeros(3)
    costD = 1.0
    cost =np.zeros(iterations,dtype=float)
    i=0
    wChanges=1.0
    # coefficients_history =np.zeros((iterations,2))
    while i<iterations:
        for col in range(3):
            costD =0.0
            for row in range(numberOfData):
                costD = costD +(( h[row] - houseData_std[row][3])*houseData_std[row][col])
                cost[i] = cost[i] + ( h[row] - houseData_std[row][3])*( h[row] - houseData_std[row][3])
            costD = costD/numberOfData
            cost[i] = cost[i]/ (2*numberOfData)
            newW[col] = coefficients[col] - alpha* costD

        wChanges = math.sqrt(pow(newW[0] - coefficients[0],2) +  pow(newW[1] - coefficients[1],2) +  pow(newW[2] - coefficients[2],2))
        # print(wChanges)
        coefficients[0] = newW[0]
        coefficients[1] = newW[1]
        coefficients[2] = newW[2]
        h = h_func(coefficients)
        i=i+1
        # if wChanges <0.0001:
        #     break
    
    # print(cost)
    # print(i)
    return cost
def plot_cost(cost):
    plt.figure(figsize=(12,8))
    plt.plot(range(iterations), cost,'b.')
    plt.xlabel("Iterations")
    plt.ylabel("J(Theta)")
    plt.savefig("GD_cost")
main()




