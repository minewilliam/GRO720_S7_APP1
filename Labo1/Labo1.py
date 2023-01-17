import time
import numpy as np
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
# mpl.use('TkAgg')

#matA = np.matrix([[3, 4, 1, 2, 1, 5],
#                  [5, 2, 3, 2, 2, 1],
#                  [6, 2, 2, 6, 4, 5],
#                  [1, 2, 1, 3, 1, 2],
#                  [1, 5, 2, 3, 3, 3],
#                 [1, 2, 2, 4, 2, 1]])

matA = np.matrix([[2, 1, 1, 2],
                  [1, 2, 3, 2],
                  [2, 1, 1, 2],
                  [3, 1, 4, 1]])

#matA = np.matrix([[3,4,1],
#                 [5,2,3],
#                  [6,2,2]])

mu = [0.001, 0.005, 0.01]
error_colors = ['r', 'g', 'b']
mat_I = np.identity(matA.shape[0])
iterations = 1000



def costfunc1(mat_a, mat_b):
    mat_BA = np.matmul(mat_b, mat_a)
    return np.linalg.matrix_power(mat_BA - mat_I, 2).sum()


def gradient1(mat_a, mat_b):
    mat_BA = np.matmul(mat_b, mat_a)
    return np.matmul(mat_BA - mat_I, mat_a.transpose()) * 2


# Average Quadratic Error
def costfunc2(y, y_actual):
    return np.power(y - y_actual, 2).sum()


def question1():
    for i, v in enumerate(mu):
        matB = np.zeros(matA.shape) # np.random.rand(matA.shape[0], matA.shape[1])
        errors = np.zeros(iterations)
        xAxisLim = iterations
        for j in range(iterations):
            matB = matB - v * gradient1(matA, matB)
            errors[j] = costfunc1(matA, matB)
            if j > 100 and errors[j] > 1000:
                xAxisLim = j
                break
        plt.plot(errors[:xAxisLim], color=error_colors[i], label="mu={0}".format(v))
        plt.xlabel("Iterations")
        plt.ylabel("error")
        plt.legend()
        plt.show()


def question2():
    polynomial = 12
    learningRate = 0.02
    exponents = np.array(range(polynomial+1))
    weights = np.array([np.zeros(polynomial+1)])
    inputs = np.array([-0.95, -0.82, -0.62, -0.43, -0.17, -0.07, 0.25, 0.38, 0.61, 0.79, 1.04, 0.12, 0.2])
    outputs = np.array([0.02, 0.03, -0.17, -0.12, -0.37, -0.25, -0.1, 0.14, 0.53, 0.71, 1.53, 2.01, 1.98])
    errors = np.zeros(iterations)
    xx = np.arange(-1.25, 1.25, 0.001)

    for epoch in range(iterations):
        grad = np.zeros(len(weights))
        for batch in range(len(inputs)):
            x = np.power(inputs[batch], exponents)
            y = np.matmul(weights, x.transpose())
            diff = y - outputs[batch]
            grad = grad + 2 * diff * x
            errors[epoch] = errors[epoch] + np.power(diff, 2)
        weights = weights - learningRate * grad

        poly = np.power(xx, np.atleast_2d(exponents).transpose())
        f = np.matmul(weights, poly)[0]
        #for index, param in enumerate(weights[0]):
        #    f = f + param * np.power(xx, index)

        if epoch % 100:
            plt.clf()
            plt.plot(xx, f)
            plt.plot(inputs, outputs, 'o')
            plt.draw()
            plt.pause(0.001)
            #plt.plot(errors, color=error_colors[0], label="mu={0}".format(learningRate))
            #plt.draw()
            #plt.pause(0.001)
    plt.show()
#question2()
question1()