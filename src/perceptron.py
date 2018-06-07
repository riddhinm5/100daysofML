import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.random.seed(42)

def getData(csv_file, x_cols, y_col):
    df = pd.read_csv(csv_file)
    X = np.array(df[x_cols])
    y = np.array(df[y_col])
    return df, X, y

def stepFunction(t):
    if t >= 0:
        return 1
    return 0


def prediction(X, W, b):
    return stepFunction((np.matmul(X, W)+b)[0])


def perceptronStep(X, y, W, b, learn_rate=0.01):
    # Fill in code
    curr_y = (np.matmul(X, W)+b).flatten()
    curr_y = [stepFunction(i) for i in curr_y]
    for i in range(len(X)):
        y_hat = prediction(X[i], W, b)
        if y[i] != y_hat:
            if y[i] == 0:
                W[0][0] -= learn_rate*X[i][0]
                W[1][0] -= learn_rate*X[i][1]
                b -= learn_rate
            else:
                W[0][0] += learn_rate*X[i][0]
                W[1][0] += learn_rate*X[i][1]
                b += learn_rate
    return W, b


def trainPerceptronAlgorithm(X, y, learn_rate=0.01, num_epochs=50):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2, 1))
    b = np.random.rand(1)[0] + x_max
    # These are the solution lines that get plotted below.
    boundary_lines = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append([-W[0]/W[1], -b/W[1]])
    return boundary_lines

def drawBoundaries(boundary_lines):
    print(boundary_lines)

def plotPerceptron(X, y, boundary_lines):
    plt.scatter(x=X[:, 0], y=X[:, 1], c=y, cmap=plt.cm.coolwarm_r, s=20)
    plt.show()


def main():
    df, X, y = getData('perceptron_data.csv', ['x1', 'x2'], 'y')
    boundary_lines = trainPerceptronAlgorithm(X, y)
    print(boundary_lines[0])
    plotPerceptron(X, y, boundary_lines)

if __name__ == '__main__':
    main()
