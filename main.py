import csv
import numpy as np
import matplotlib.pyplot as plt
import random

filename="transport_data.csv"
with open(filename,"r",newline="") as file:
    reader=csv.reader(file)
    rownum=0
    a=[]
    next(file)
    for row in reader:
        a.append(row)
        rownum+=1

a = np.delete(a,2,1)

max_longitude = max(a[:,0])
max_latitude = max(a[:,1])
max_time = max(a[:,2])
min_longitude = min(a[:,0])
min_latitude = min(a[:,1])
min_time = min(a[:,2])

train = []
test = []

for i in range(len(a)):
        if a[i][3] == '?':
            test.append(a[i])
        elif a[i][3] != '-':
            train.append(a[i])

x_train = np.array(train)[:, 0:3]
y_train_category = np.array(train)[:, 3]
y_train_size = len(y_train_category)
y_train = np.zeros((y_train_size, 3))

for i in range(len(y_train_category)):
    ind = y_train_category[i]
    ind = ind.astype(np.int)
    y_train[i][ind] = 1
print(y_train[9])
print(y_train_category[9])

x_test = np.array(test)[:, 0:3]
y_test = []

x_train = x_train.astype(np.float)

x_test = x_test.astype(np.float)

#x_train[0] = np.divide(x_train[0], max_longitude)


max_longitude = max_longitude.astype(np.float)
max_latitude = max_latitude.astype(np.float)
max_time = max_time.astype(np.float)
min_longitude = min_longitude.astype(np.float)
min_latitude = min_latitude.astype(np.float)
min_time = min_time.astype(np.float)

delta_longitude = max_longitude - min_longitude
delta_latitude = max_latitude - min_latitude
delta_time = max_time - min_time

#normalization ~ [0,1]
for i in range(len(x_train)):
        x_train[i][0] = (x_train[i][0]-min_longitude)/delta_longitude
        x_train[i][1] = (x_train[i][1]-min_latitude)/delta_latitude
        x_train[i][2] = (x_train[i][2]-min_time)/delta_time
for i in range(len(x_test)):
        x_test[i][0] = (x_test[i][0]-min_longitude)/delta_longitude
        x_test[i][1] = (x_test[i][1]-min_latitude)/delta_latitude
        x_test[i][2] = (x_test[i][2]-min_time)/delta_time

layers_number = 3

w1 = np.array([
  [0.3, 0.2, 0.1],
  [0.3, 0.6, 0.5],
  [0.1, 0.1, 0.4]
])
w2 = np.array([
  [0.4, 0.3, 0.1],
  [0.6, 0.2, 0.6],
  [0.2, 0.1, 0.3]
])
for i in range(len(w1)):
    for j in range(len(w1[i])):
        w1[i][j] = random.uniform(-0.5,0.5)
        
for i in range(len(w2)):
    for j in range(len(w2[i])):
        w2[i][j] = random.uniform(-0.5,0.5)

for i in range(len(w2)):
    for j in range(len(w2[i])):
        print(w2[i][j])
        
#w3 = np.array([0.2,0.4,0.6])

def activation_func(x):
    fx = 1/(1+np.exp(-x))
    return fx

def activation_func_der(x):
    dfx = activation_func(x)*(1-activation_func(x))
    return dfx

def calculate_y(x, w1, w2):
    #print(x)
    z = [0] * (len(x))
    for i in range(len(z)):
        for j in range(len(x)):
            #print(x[j]*w1[j][i])
            z[i]=z[i]+x[j]*w1[j][i]
        z[i]=activation_func(z[i])
    #print(z)
    y = [0] * (len(z))
    for i in range(len(y)):
        for j in range(len(z)):
            y[i]=y[i]+z[j]*w2[j][i]
        y[i]=activation_func(y[i])
    return y

#learning
alpha = 0.5 #learning speed
calc_error = 0
n = 0
while n < len(x_train):
    calc_error = 0
    y = calculate_y(x_train[n],w1,w2)
    for j in range(len(y_train[0])):
        calc_error = calc_error + (y[j] - y_train[j])
        

#calculating
for j in range(len(x_test)):
    y = calculate_y(x_test[j],w1,w2)
    y_max_index = y.index(max(y))
    y_test.append(y_max_index)

with open("result.txt","w",newline="") as file:
    for i in range(len(y_test)):
        file.write(str(y_test[i]))
        file.write("\n")
