import csv
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.optimize import minimize

filename="transport_data.csv"
print("Importing data from input file..")
with open(filename,"r",newline="") as file:
    reader=csv.reader(file)
    rownum=0
    a=[]
    next(file)
    for row in reader:
        a.append(row)
        rownum+=1
print("Imported.")
a = np.delete(a,2,1)

max_longitude = max(a[:,0])
max_latitude = max(a[:,1])
max_time = max(a[:,2])
min_longitude = min(a[:,0])
min_latitude = min(a[:,1])
min_time = min(a[:,2])

known = []
train = []
test = []
task = []

print("Dividing data...")
for i in range(len(a)):
        if a[i][3] == '?':
            task.append(a[i])
        elif a[i][3] != '-': # 40 000 records
            known.append(a[i])

training_percent = 0.1
record_number = len(known)
test = np.array(known)[int(record_number * training_percent) : record_number]
train = np.array(known)[0:int(record_number * training_percent)]

x_train = np.array(train)[:, 0:3]
y_train = np.array(train)[:, 3]

x_test = np.array(test)[:, 0:3]
y_test = np.array(test)[:, 3]

x_task = np.array(task)[:, 0:3]
y_task = []

x_train = x_train.astype(np.float)
x_test = x_test.astype(np.float)
x_task = x_task.astype(np.float)
y_train = y_train.astype(int)
y_test = y_test.astype(int)
print("Divided.")

print("Normalizing data...")
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
for i in range(len(x_task)):
        x_task[i][0] = (x_task[i][0]-min_longitude)/delta_longitude
        x_task[i][1] = (x_task[i][1]-min_latitude)/delta_latitude
        x_task[i][2] = (x_task[i][2]-min_time)/delta_time
print("Normalized.")
x_first_train = x_train
y_first_train = y_train

for i in range(len(y_first_train)):
    if y_train[i] == '0':
        y_first_train[i] = 1
    else:
        y_first_train[i] = -1

y_first_train = y_first_train.astype(int)

print("Building dividing hyperplain...")

n = len(train)
#0<=lambdai<=c, 1<=i<=n
#summ(i=1...n)lambdai*ci=0
def func(lmbda):
    result = 0
    for i in range(n):
        for j in range(n):
            temp = lmbda[i]*lmbda[j]
            temp = temp * int(y_first_train[i])
            temp = temp * int(y_first_train[j])
            x_dot = np.dot(x_first_train[i],x_first_train[j])
            temp = temp * x_dot
            result = result + temp
            #result=result+lmbda[i]*lmbda[j]*y_first_train[i]*y_first_train[j]*np.dot(x_first_train[i],x_first_train[j])
    result = result/2
    for i in range(n):
        result = result - lmbda[i]
    return result

def func_deriv(lmbda):
    result = [-1] * len(lmbda)
    for i in range(len(lmbda)):
        #result[i] = -1
        for j in range(n):
            if j != i:
                temp = lmbda[j]
                temp = temp * int(y_first_train[i])
                temp = temp * int(y_first_train[j])
                x_dot = np.dot(x_first_train[i],x_first_train[j])
                temp = temp * x_dot
                temp = temp / 2
                result[i] = result[i] + temp
        result[i] = result[i] + int(y_first_train[i])*int(y_first_train[i])*np.dot(x_first_train[i],x_first_train[i])
            #result=result+lmbda[i]*lmbda[j]*y_first_train[i]*y_first_train[j]*np.dot(x_first_train[i],x_first_train[j])
    return result

#learning
w = [0] * len(x_first_train[0])
b = 0
C = 1
lambda0 = [0] * n #initial point
print("Initialazing bounds & constraints...")
bnds = [[0,C]] * n
print(len(y_first_train))

#cons = ({'type' : 'eq',
#         'fun' : lambda lmbda : np.dot(lmbda,y_first_train),
#         'jac' : lambda lmbda : np.sum(y_first_train)})
cons = ({'type' : 'eq',
         'fun' : lambda lmbda : np.dot(lmbda,y_first_train)})
#res = minimize(func, lambda0, method = 'SLSQP', bounds = bnds, constraints = cons)

print("Minimizing Lagrangian...")

res = minimize(func, lambda0, jac = func_deriv, method = 'SLSQP', bounds = bnds, constraints = cons)
opt_lambda = res.x

print("Minimized.")
print("Calculating wx+b=0...")
print("Calculating w...")
for i in range(n):
    for j in range(len(x_first_train[0])):
        w[j] = w[j] + opt_lambda[i]*int(y_first_train[i])*x_first_train[i][j]
print(w)
print("Calculated.")
print("Calculating b...")

b = np.dot(w,x_first_train[0])
b = b - int(y_first_train[0])
#for i in range(len(x_first_train[0])):
#    b[i] = w[i]*x_first_train[0][i] - int(y_first_train[0])
print(b)
print("Calculated.")

print("Classifying data...")

def classify(x):
    return np.sign(np.dot(w,x)+b)

y_classified = []
for j in range(len(x_test)):
    x_class = classify(x_test[i])
    y_classified.append(x_class)
print("Classified.")

print("Writing data to output file...")

with open("result.txt","w",newline="") as file:
    for i in range(len(x_test)):
        file.write(str(y_classified[i]))
        file.write("\t")
        file.write(str(y_test[i]))
        file.write("\n")
print("Finished.")

