# Machine Learning HW1

import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
import scipy as sp
import matplotlib.pyplot as plt
import random
from scipy import stats
from scipy.optimize import fmin
# more imports

# Parse the file and return 2 numpy arrays
def load_data_set(filename):
    x = np.loadtxt(filename, usecols=(0,1))
    y = np.loadtxt(filename, usecols=2)
    return x, y

# Find theta using the normal equation
def normal_equation(x, y):
    theta = np.linalg.inv(x.T.dot(x)).dot(x.T.dot(y))
    return theta

# Find thetas using stochastic gradient descent
# Don't forget to shuffle
def stochastic_gradient_descent(x, y, learning_rate, num_epoch):
    thetas = minibatch_gradient_descent(x, y, learning_rate, num_epoch, 1)
    return thetas

# Find thetas using gradient descent
def gradient_descent(x, y, learning_rate, num_epoch):
    theta = [1,1]
    thetas = []
    for i in range(num_epoch):
        loss = np.dot(x, theta) - y
        gradient = np.dot(x.T,loss)
        gradient = gradient/len(x)
        theta = theta - learning_rate*gradient
        thetas.append(theta)
    return thetas

# Find thetas using minibatch gradient descent
# Don't forget to shuffle
def minibatch_gradient_descent(x, y, learning_rate, num_epoch, batch_size):
    theta = [1,1]
    thetas = []
    for epoch in range(num_epoch):
        randomize = np.arange(len(x))
        np.random.shuffle(randomize)
        x = x[randomize]
        y = y[randomize]

        for batch in range(0, len(x), batch_size):
            batch_x, batch_y = x[batch:batch + batch_size], y[batch:batch + batch_size]
            loss = batch_y - np.dot(batch_x,theta)
            gradient = np.dot(batch_x.T, loss)
            theta = (theta + (learning_rate*(gradient/batch_size)))
        thetas.append(theta)

    return thetas

# Given an array of x and theta predict y
def predict(x, theta):
   y_predict = np.dot(x,theta)
   return y_predict

# Given an array of y and y_predict return MSE loss
def get_mseloss(y, y_predict):
    loss = 0
    loss = np.dot((y-y_predict),((y-y_predict).T))
    loss = loss / len(y)
    return loss

# Given a list of thetas one per epoch
# this creates a plot of epoch vs training error
def plot_training_errors(x, y, thetas, title):
    losses = []
    epochs = []
    losses = []
    epoch_num = 1
    for theta in thetas:
        losses.append(get_mseloss(y, predict(x, theta)))
        epochs.append(epoch_num)
        epoch_num += 1
    plt.plot(epochs, losses)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(title)
    plt.show()

# Given x, y, y_predict and title,
# this creates a plot
def plot(x, y, theta, title):
    y_predict = predict(x, theta)
    plt.scatter(x[:, 1], y)
    slope = round(theta[0], 4)
    intercept = round(theta[1], 4)
    plt.plot(x[:, 1], y_predict, label =  str(intercept) + 'x' + '+' + str(slope))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend(loc='upper left')
    plt.show()


if __name__ == "__main__":
    x, y = load_data_set('regression-data.txt')
    plot
    plt.scatter(x[:, 1], y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Scatter Plot of Data")
    plt.show()

    theta = normal_equation(x, y)
    plot(x, y, theta, "Normal Equation Best Fit")

    thetas = gradient_descent(x, y, 0.001, 100) 
    plot(x, y, thetas[-1], "Gradient Descent Best Fit (Î± = 0.001)")
    plot_training_errors(x, y, thetas, "Gradient Descent Epoch vs Mean Training Loss (Î± = 0.001)" )

    thetas = gradient_descent(x, y, 0.005, 100) 
    plot(x, y, thetas[-1], "Gradient Descent Best Fit (Î± = 0.005)")
    plot_training_errors(x, y, thetas, "Gradient Descent Epoch vs Mean Training Loss (Î± = 0.005)")

    thetas = gradient_descent(x, y, 0.008, 100) 
    plot(x, y, thetas[-1], "Gradient Descent Best Fit (Î± = 0.008)")
    plot_training_errors(x, y, thetas, "Gradient Descent Epoch vs Mean Training Loss (Î± = 0.008)")

    thetas = gradient_descent(x, y, 0.01, 100) 
    plot(x, y, thetas[-1], "Gradient Descent Best Fit (Î± = 0.01)")
    plot_training_errors(x, y, thetas, "Gradient Descent Epoch vs Mean Training Loss (Î± = 0.01)")

    thetas = gradient_descent(x, y, 0.03, 100) 
    plot(x, y, thetas[-1], "Gradient Descent Best Fit (Î± = 0.03)")
    plot_training_errors(x, y, thetas, "Gradient Descent Epoch vs Mean Training Loss (Î± = 0.03)")

    thetas = gradient_descent(x, y, 0.1, 100) 
    plot(x, y, thetas[-1], "Gradient Descent Best Fit (Î± = 0.1)")
    plot_training_errors(x, y, thetas, "Gradient Descent Epoch vs Mean Training Loss (Î± = 0.1)")

    thetas = gradient_descent(x, y, 0.2, 100) 
    plot(x, y, thetas[-1], "Gradient Descent Best Fit (Î± = 0.2)")
    plot_training_errors(x, y, thetas, "Gradient Descent Epoch vs Mean Training Loss (Î± = 0.2)")

    thetas = gradient_descent(x, y, 0.3, 100) 
    plot(x, y, thetas[-1], "Gradient Descent Best Fit (Î± = 0.3)")
    plot_training_errors(x, y, thetas, "Gradient Descent Epoch vs Mean Training Loss (Î± = 0.3)")
    
    thetas = gradient_descent(x, y, 0.6, 100) 
    plot(x, y, thetas[-1], "Gradient Descent Best Fit (Î± = 0.6)")
    plot_training_errors(x, y, thetas, "Gradient Descent Epoch vs Mean Training Loss (Î± = 0.6)")

    # You should try multiple non-zero learning rates and  multiple different (non-zero) number of epoch
    thetas = stochastic_gradient_descent(x, y, 0.00008, 100) # Try different learning rates and number of epoch
    plot(x, y, thetas[-1], "stochastic Gradient Descent Best Fit (Î± = 0.00008)")
    plot_training_errors(x, y, thetas, "Stochastic Gradient Descent Epoch vs Mean Training Loss (Î± = 0.00008)")

    thetas = stochastic_gradient_descent(x, y, 0.001, 100) # Try different learning rates and number of epoch
    plot(x, y, thetas[-1], "stochastic Gradient Descent Best Fit (Î± = 0.001)")
    plot_training_errors(x, y, thetas, "Stochastic Gradient Descent Epoch vs Mean Training Loss (Î± = 0.001)")

    thetas = stochastic_gradient_descent(x, y, 0.005, 100) # Try different learning rates and number of epoch
    plot(x, y, thetas[-1], "stochastic Gradient Descent Best Fit (Î± = 0.005)")
    plot_training_errors(x, y, thetas, "Stochastic Gradient Descent Epoch vs Mean Training Loss (Î± = 0.005)")

    thetas = stochastic_gradient_descent(x, y, 0.01, 100) # Try different learning rates and number of epoch
    plot(x, y, thetas[-1], "stochastic Gradient Descent Best Fit (Î± = 0.01)")
    plot_training_errors(x, y, thetas, "Stochastic Gradient Descent Epoch vs Mean Training Loss (Î± = 0.01)")

    thetas = stochastic_gradient_descent(x, y, 0.05, 100) # Try different learning rates and number of epoch
    plot(x, y, thetas[-1], "stochastic Gradient Descent Best Fit (Î± = 0.05)")
    plot_training_errors(x, y, thetas, "Stochastic Gradient Descent Epoch vs Mean Training Loss (Î± = 0.05)")

    thetas = stochastic_gradient_descent(x, y, 0.1, 100) # Try different learning rates and number of epoch
    plot(x, y, thetas[-1], "stochastic Gradient Descent Best Fit (Î± = 0.1)")
    plot_training_errors(x, y, thetas, "Stochastic Gradient Descent Epoch vs Mean Training Loss (Î± = 0.1)")

    thetas = stochastic_gradient_descent(x, y, 0.3, 100) # Try different learning rates and number of epoch
    plot(x, y, thetas[-1], "stochastic Gradient Descent Best Fit (Î± = 0.3)")
    plot_training_errors(x, y, thetas, "Stochastic Gradient Descent Epoch vs Mean Training Loss (Î± = 0.3)")

    thetas = stochastic_gradient_descent(x, y, 0.6, 100) # Try different learning rates and number of epoch
    plot(x, y, thetas[-1], "stochastic Gradient Descent Best Fit (Î± = 0.6)")
    plot_training_errors(x, y, thetas, "Stochastic Gradient Descent Epoch vs Mean Training Loss (Î± = 0.6)")


    # You should try multiple non-zero learning rates and  multiple different (non-zero) number of epoch
    thetas = minibatch_gradient_descent(x, y, 0.0001, 100, 20)
    plot(x, y, thetas[-1], "Minibatch Gradient Descent Best Fit (Î± = 0.0001, batch size = 20)")
    plot_training_errors(x, y, thetas, "Minibatch Gradient Descent Epoch vs Mean Training Loss (Î± = 0.0001, batch size = 20)")

    thetas = minibatch_gradient_descent(x, y, 0.001, 100, 20)
    plot(x, y, thetas[-1], "Minibatch Gradient Descent Best Fit (Î± = 0.001, batch size = 20)")
    plot_training_errors(x, y, thetas, "Minibatch Gradient Descent Epoch vs Mean Training Loss (Î± = 0.001, batch size = 20)")

    thetas = minibatch_gradient_descent(x, y, 0.005, 100, 20)
    plot(x, y, thetas[-1], "Minibatch Gradient Descent Best Fit (Î± = 0.005, batch size = 20)")
    plot_training_errors(x, y, thetas, "Minibatch Gradient Descent Epoch vs Mean Training Loss (Î± = 0.005, batch size = 20)")

    thetas = minibatch_gradient_descent(x, y, 0.01, 100, 20)
    plot(x, y, thetas[-1], "Minibatch Gradient Descent Best Fit (Î± = 0.01, batch size = 20)")
    plot_training_errors(x, y, thetas, "Minibatch Gradient Descent Epoch vs Mean Training Loss (Î± = 0.01, batch size = 20)")
    
    thetas = minibatch_gradient_descent(x, y, 0.02, 100, 20)
    plot(x, y, thetas[-1], "Minibatch Gradient Descent Best Fit (Î± = 0.02, batch size = 20)")
    plot_training_errors(x, y, thetas, "Minibatch Gradient Descent Epoch vs Mean Training Loss (Î± = 0.02, batch size = 20)")

    thetas = minibatch_gradient_descent(x, y, 0.02, 100, 5)
    plot(x, y, thetas[-1], "Minibatch Gradient Descent Best Fit (Î± = 0.02, batch size = 5)")
    plot_training_errors(x, y, thetas, "Minibatch Gradient Descent Epoch vs Mean Training Loss (Î± = 0.02, batch size = 5)")

    thetas = minibatch_gradient_descent(x, y, 0.02, 100, 40)
    plot(x, y, thetas[-1], "Minibatch Gradient Descent Best Fit (Î± = 0.02, batch size = 40)")
    plot_training_errors(x, y, thetas, "Minibatch Gradient Descent Epoch vs Mean Training Loss (Î± = 0.02, batch size = 40)")

    thetas = minibatch_gradient_descent(x, y, 0.02, 100, 100)
    plot(x, y, thetas[-1], "Minibatch Gradient Descent Best Fit (Î± = 0.02, batch size = 100)")
    plot_training_errors(x, y, thetas, "Minibatch Gradient Descent Epoch vs Mean Training Loss (Î± = 0.02, batch size = 100)")

    thetas = minibatch_gradient_descent(x, y, 0.05, 100, 20)
    plot(x, y, thetas[-1], "Minibatch Gradient Descent Best Fit (Î± = 0.05, batch size = 20)")
    plot_training_errors(x, y, thetas, "Minibatch Gradient Descent Epoch vs Mean Training Loss (Î± = 0.05, batch size = 20)")

    thetas = minibatch_gradient_descent(x, y, 0.1, 100, 20)
    plot(x, y, thetas[-1], "Minibatch Gradient Descent Best Fit (Î± = 0.1, batch size = 20)")
    plot_training_errors(x, y, thetas, "Minibatch Gradient Descent Epoch vs Mean Training Loss (Î± = 0.1, batch size = 20)")

    thetas = minibatch_gradient_descent(x, y, 0.3, 100, 20)
    plot(x, y, thetas[-1], "Minibatch Gradient Descent Best Fit (Î± = 0.3, batch size = 20)")
    plot_training_errors(x, y, thetas, "Minibatch Gradient Descent Epoch vs Mean Training Loss (Î± = 0.3, batch size = 20)")

    thetas = minibatch_gradient_descent(x, y, 0.001, 100, 5)
    plot(x, y, thetas[-1], "Minibatch Gradient Descent Best Fit (Î± = 0.001, batch size = 5)")
    plot_training_errors(x, y, thetas, "Minibatch Gradient Descent Epoch vs Mean Training Loss (Î± = 0.001, batch size = 5)")

    thetas = minibatch_gradient_descent(x, y, 0.01, 100, 40)
    plot(x, y, thetas[-1], "Minibatch Gradient Descent Best Fit (Î± = 0.01, batch size = 40)")
    plot_training_errors(x, y, thetas, "Minibatch Gradient Descent Epoch vs Mean Training Loss (Î± = 0.01, batch size = 40)")

    thetas = minibatch_gradient_descent(x, y, 0.1, 100, 100)
    plot(x, y, thetas[-1], "Minibatch Gradient Descent Best Fit (Î± = 0.1, batch size = 100)")
    plot_training_errors(x, y, thetas, "Minibatch Gradient Descent Epoch vs Mean Training Loss (Î± = 0.1, batch size = 100)")