import numpy as np


def softmax(z):
    '''
    :param z:
    :return:
    '''
    z -= np.max(z, axis=1).reshape(-1, 1)
    s = (np.exp(z) / np.sum(np.exp(z), axis=1).reshape(-1, 1))
    return s


def initialize(X, y):
    '''
    :param X: features
    :param y: labels
    :return: weights, base
    '''
    weights = np.zeros((X.shape[1], y.shape[1]))
    base = np.zeros((1, y.shape[1]))
    return weights, base


def propagate(weights, base, X, y):
    '''
    :param weights:
    :param base:
    :param X: features
    :param y: labels
    :return: cost, gradient_weights, gradient_base
    '''
    m = X.shape[0]
    h = softmax(np.dot(X, weights) + base)
    cost = (-1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    grad_w = (1/m) * np.dot(X.T, (h - y))
    grad_b = (1/m) * np.sum((h-y), axis=0)
    return cost, grad_w, grad_b


def predict(X, weights, base):
    '''
    :param X: features
    :param weights:
    :param base:
    :return: predicted labels
    '''
    h = softmax(np.dot(X, weights) + base)
    return h


def evaluate(X, y, weights, base):
    '''
    :param X: features
    :param y: labels
    :param weights:
    :param base:
    :return: accuracy
    '''
    h = softmax(np.dot(X, weights) + base)
    h_argmax = np.argmax(h, axis=1)
    y_argmax = np.argmax(y, axis=1)
    accuracy = sum(h_argmax == y_argmax)/(float(len(y)))
    return accuracy


def model(train_x, train_y, test_x, test_y, iters, alpha, print_cost=True):
    '''
    :param train_x: training features
    :param train_y: training labels
    :param test_x: test features
    :param test_y: test labels
    :param iters: no. of iterations
    :param alpha: learning rate
    :param print_cost: printing the cost
    :return: weights, base
    '''
    # reshaping (28, 28) data int 784
    train_x = np.reshape(train_x, (-1, 784))
    test_x = np.reshape(test_x, (-1, 784))

    # initializing weights and bias
    weights, base = initialize(train_x, train_y)
    print("\nTraining multiclass Logistic Regression on MNIST data...")
    for i in range(iters):
        # getting the cost, weight gradient and base gradient
        cost, grad_w, grad_b = propagate(weights, base, train_x, train_y)
        weights = weights - alpha * grad_w
        base = base - alpha * grad_b
        if print_cost and i % 100 == 0:
            print('cost after iteration {} : {}'.format(i, cost))
    print(f'Cost after iteration {i+1} : {cost}')

    # calculating train and test accuracy
    train_accuracy = evaluate(train_x, train_y, weights, base)
    test_accuracy = evaluate(test_x, test_y, weights, base)
    train_size = len(train_y)
    print(f'Logistic Regression Train accuracy : {train_accuracy}%')
    print(f'Logistic Regression Test accuracy : {test_accuracy}%')
    print(f'Training size : {train_size}, alpha : {alpha}, iterations : {iters}\n\n')
    LR_params = {'weights': weights, 'base': base, 'train_accuracy': train_accuracy,
                 'test_accuracy': test_accuracy, 'train_size': train_size, 'alpha': alpha, 'iters': iters}
    return LR_params
