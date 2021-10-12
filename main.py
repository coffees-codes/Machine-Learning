import numpy as np
import matplotlib.pyplot as plt


def featureNormalize(X):
    mu = np.zeros((1, np.size(X, 1)))
    sigma = np.zeros((1, np.size(X, 1)))
    X_norm = np.divide((X - np.mean(X, axis=0)), (np.std(X, axis=0)))
    return list((X_norm, mu, sigma))


def computeCost(X, y, theta):
    m = len(y)
    term = np.dot(X, theta) - y
    J = np.dot(term.T, term)/(2*m)
    return J[0][0]


def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros((num_iters, 1))
    for i in range(num_iters):
        theta = theta - (alpha/m) * (np.sum(np.multiply(np.sum(np.multiply(X,theta.T), axis=1).reshape(m,1) - y, X), axis=0).reshape(np.shape(theta)))
        J_history[i] = computeCost(X, y, theta)
    return list((theta, J_history))


file = open("data.csv")
numpy_array = np.loadtxt(file, delimiter=",")
m = np.shape(numpy_array)[0]    # number of training examples
n = np.shape(numpy_array)[1] - 1
X = np.array(numpy_array[:, 0:n].reshape(m, n))
y = np.array(numpy_array[:, n].reshape(m, 1))

if n > 1:
    X, mu, sigma = featureNormalize(X)  # feature normalization

X = np.append(np.ones((1, m)).T, X, axis=1)     # adding bias feature
theta, temp = gradientDescent(X, y, theta=np.zeros((n+1, 1)), alpha=0.09, num_iters=400)
if n == 1:
    plt.scatter(X[:, -1], y, marker='x', color='r')
    plt.plot(X[:, -1], np.dot(X, theta))
else:
    plt.plot(range(50), temp[0:50])

plt.show()
l = []
for i in range(n):
    t = int(input("Enter value of feature "+str(i+1)+": "))
    l.append(t)
if n>1:
    l = np.divide((l - np.mean(l, axis=0)), (np.std(l, axis=0)))
l = np.insert(l, 0, 1)
l = np.array(l)


print("Prediction of price: ", np.dot(l, theta)[0])



