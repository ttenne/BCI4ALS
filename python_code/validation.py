import SVM
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import csv

def validateNumOfLags(min_lags=15, max_lags=25):
    accuracies = []
    lags_numbers = list(range(min_lags, max_lags, 1))
    for lags in lags_numbers:
        print(f'Running lags={lags}')
        y_pred, y_test = SVM.svmPredict(lags=lags)
        accuracies.append(SVM.accuracy(y_test, y_pred))
    plt.plot(lags_numbers, accuracies)
    plt.xlabel('# of lags')
    plt.ylabel('validation score')
    plt.savefig('validateNumOfLags.png')
    print(f'accuracies = {accuracies}')
    print(f'max accuracy = {np.max(accuracies)}')
    print(f'# of lags = {lags_numbers[np.argmax(accuracies)]}')

def validateLagsStartingPoint(min_start_point=100, max_start_point=150):
    accuracies = []
    starting_points = list(range(min_start_point, max_start_point, 5))
    for start in starting_points:
        print(f'Running start={start}')
        y_pred, y_test = SVM.svmPredict(lags_starting_point=start)
        accuracies.append(SVM.accuracy(y_test, y_pred))
    plt.plot(starting_points, accuracies)
    plt.xlabel('starting point')
    plt.ylabel('validation score')
    plt.savefig('validateLagsStartingPoint.png')
    print(f'accuracies = {accuracies}')
    print(f'max accuracy = {np.max(accuracies)}')
    print(f'starting point = {starting_points[np.argmax(accuracies)]}')

def saveResultsInCsv(results, x, y):
    with open('validateAR.csv', mode='w') as fp:
        writer = csv.writer(fp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['']+x)
        results = [np.insert(row, 0, y[i]) for i, row in enumerate(results)]
        writer.writerows(results)

def validateAR(min_lags=18, max_lags=23, min_start_point=115, max_start_point=150):
    '''2-D validation for both hyper-parameters of the AR Model'''
    lags_numbers = list(range(min_lags, max_lags, 1))
    starting_points = list(range(min_start_point, max_start_point, 5))
    accuracies = np.zeros((len(starting_points),len(lags_numbers)))
    for i, start in enumerate(starting_points):
        for j, lags in enumerate(lags_numbers):
            print(f'Running start={start}, lags={lags}')
            try:
                y_pred, y_test = SVM.svmPredict(lags=lags, lags_starting_point=start)
            except ZeroDivisionError:
                #this means we got to the maximal lags number for current starting point
                break
            except ValueError:
                #this means we got to the maximal lags number for current starting point
                break
            else:
                accuracies[i][j] = SVM.accuracy(y_test, y_pred)
    print(accuracies)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    X, Y = np.meshgrid(lags_numbers, starting_points)
    surf = ax.plot_surface(X, Y, accuracies, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel('lags')
    ax.set_ylabel('starting point')
    ax.set_zlabel('accuracy')
    saveResultsInCsv(accuracies, lags_numbers, starting_points)
    plt.show()

validateNumOfLags()
# validateLagsStartingPoint()
# validateAR()
