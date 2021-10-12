import numpy as np
import pandas as pd
import sys

def sigmoid(z):
    g = np.divide(np.ones(np.shape(z)), 1 + np.exp(-z))
    return g


def costFunction(theta, X, y):
    m = len(y)
    J = (np.sum(np.multiply(y, np.log(sigmoid(np.dot(X, theta)))) + np.multiply(1 - y, np.log(1 - sigmoid(np.dot(X, theta)))), axis=0))/-m
    grad = np.dot(X.T, sigmoid(np.dot(X, theta)) - y)/m
    return list((J, grad))


def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    # J_history = np.zeros((num_iters, 1))
    for i in range(num_iters):
        theta = theta - (alpha/m) * (np.dot(X.T, sigmoid(np.dot(X, theta)) - y))
        # J_history[i] = costFunction(theta, X, y)[0]
    return theta


def pred(theta, features):
    prob = sigmoid(np.dot(features, theta))
    if prob > 0.5:
        return 1
    else:
        return 0


d = pd.read_csv('titanic.csv')  # reading file
m = d.shape[0]
y = [d['Survived']]
y = np.array(y).reshape(m, 1)   # initializing y

# replacing 'Sex' values with 1 for male and 0 for female:
m_indices = [i for i in range(m) if d['Sex'][i] == 'male']
temp = []
for i in range(m):
    if i in m_indices:
        temp.append(1)
    else:
        temp.append(0)
d['Sex'] = temp

# initializing feature matrix X
X = [d[i] for i in d if not i == 'Name']
X.pop(0)
X = np.array(X)
X = X.T
n = np.shape(X)[1]
initial_theta = np.zeros((n+1, 1))  # adding bias feature

X = np.append(np.ones((1, m)).T, X, axis=1)
cost, grad = costFunction(initial_theta, X, y)
theta = gradientDescent(X, y, initial_theta, alpha=0.004, num_iters=400)
cost = costFunction(theta, X, y)[0]
# print(theta)

feature_ip = [1]
feature_heads = ['Pclass', 'Sex', 'Age', 'Siblings/Spouses Aboard', 'Parents/Children Aboard', 'Fare']
for i in feature_heads:
    if i == 'Sex':
        s = input("Enter M for male or F for female:")
        if s in ['M', 'm']:
            feature_ip.append(1)
        elif s in ['F', 'f']:
            feature_ip.append(0)
        else:
            print('Invalid input')
            sys.exit()
    else:
        feature_ip.append(float(input("Enter "+i)))

feature_ip = np.array(feature_ip)
p = pred(theta, feature_ip)
if p:
    print("Prediction: Survived")
else:
    print("Prediction: Did not survive")
