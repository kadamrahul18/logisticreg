import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def risk(X, Y, theta):
    f = 1 / (1 + np.exp(-np.dot(X, theta)))
    r = (Y - 1) * np.log(1 - f) - Y * np.log(f)
    r[np.isnan(r)] = 0
    return np.mean(r)

def gradient(X, Y, theta):
    f = 1 / (1 + np.exp(-np.dot(X, theta)))
    d = X * np.exp(-np.dot(X, theta))
    g = (1 - Y) * (X - d * f) - Y * d * f
    return np.mean(g, axis=0).reshape(-1, 1)

def problem4():

    # Load data from CSV file using pandas
    data = pd.read_csv(r"/Users/sunilkadam/Desktop/ML_HW/HW_1/Problem/dataset4_csv.csv", dtype=float)
    X = data.iloc[:, :-1].values  # Features (all columns except the last one)
    Y = data.iloc[:, -1].values  # Labels (the last column)
    Y = Y.reshape(-1, 1)

    stepsize = 0.7
    tol = 0.003
    theta = np.random.rand(X.shape[1], 1)
    maxiter = 200000
    curiter = 0

    risks = []
    errs = []
    prevtheta = theta + 2 * tol

    while np.linalg.norm(theta - prevtheta) >= tol:
        if curiter > maxiter:
            break

        r = risk(X, Y, theta)
        f = 1 / (1 + np.exp(-np.dot(X, theta)))
        f[f >= 0.5] = 1
        f[f < 0.5] = 0
        err = np.sum(f != Y) / len(Y)

        print(f'Iter: {curiter}, error: {err:.4f}, risk: {r:.4f}')
        risks.append(r)
        errs.append(err)

        prevtheta = np.copy(theta)
        G = gradient(X, Y, theta)
        theta -= stepsize * G

        curiter += 1

    plt.figure()
    plt.plot(range(1, curiter + 1), errs, 'blue', range(1, curiter + 1), risks, 'black')
    plt.title('Error (blue) and risk (black) vs. iterations')

    print('theta')
    print(theta)

    x = np.arange(0, 1.01, 0.01)
    y = (-theta[2] - theta[0] * x) / theta[1]

    plt.figure()
    plt.plot(x, y, 'green')
    plt.plot(X[Y.squeeze() == 0, 0], X[Y.squeeze() == 0, 1], 'r.')
    plt.plot(X[Y.squeeze() == 1, 0], X[Y.squeeze() == 1, 1], 'b.')
    plt.title('Linear decision boundary')
    plt.show()

if __name__ == '__main__':
    problem4()