import svm
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import csv

def validateNumOfLags(max_lags=626):
    accuracies = []
    for lags in range(max_lags+1):
        print(f'Running lags={lags}')
        y_pred, y_test = svm.svmPredict(lags=lags)
        accuracies.append(svm.accuracy(y_test, y_pred))
    plt.plot(accuracies)
    plt.xlabel('# of lags')
    plt.ylabel('validation score')
    plt.savefig('validateNumOfLags.png')
    print(f'accuracies = {accuracies}')
    print(f'max accuracy = {np.max(accuracies)}')
    print(f'# of lags = {np.argmax(accuracies)}')

def validateLagsStartingPoint(max_start_point=626):
    accuracies = []
    for start in range(max_start_point+1):
        print(f'Running start={start}')
        y_pred, y_test = svm.svmPredict(lags_starting_point=start)
        accuracies.append(svm.accuracy(y_test, y_pred))
    plt.plot(accuracies)
    plt.xlabel('# of lags')
    plt.ylabel('validation score')
    plt.savefig('validateLagsStartingPoint.png')
    print(f'accuracies = {accuracies}')
    print(f'max accuracy = {np.max(accuracies)}')
    print(f'# of lags = {np.argmax(accuracies)}')

def saveResultsInCsv(results):
    with open('validateAR.csv', mode='w') as fp:
        writer = csv.writer(fp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerows(results)

def validateAR(max_iter=626):
    '''2-D validation for both hyper-parameters of the AR Model'''
    accuracies = np.zeros((max_iter+1,max_iter+1))
    for start in range(max_iter+1):
        for lags in range(max_iter+1):
            print(f'Running start={start}, lags={lags}')
            try:
                y_pred, y_test = svm.svmPredict(lags=lags, lags_starting_point=start)
            except ValueError:
                pass
            else:
                accuracies[start][lags] = svm.accuracy(y_test, y_pred)
    print(accuracies)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    x = list(range(max_iter+1)) # = y
    X, Y = np.meshgrid(x, x)
    surf = ax.plot_surface(X, Y, accuracies, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel('lags')
    ax.set_ylabel('starting point')
    ax.set_zlabel('accuracy')
    saveResultsInCsv(accuracies)
    plt.show()

# validateNumOfLags()
# validateLagsStartingPoint()
validateAR()
