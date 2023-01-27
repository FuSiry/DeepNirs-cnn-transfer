import time
from sklearn.preprocessing import scale,MinMaxScaler,Normalizer,StandardScaler
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

def huber(y_true, y_pred, delta=1.0):
    y_true = y_true.reshape(-1,1)
    y_pred = y_pred.reshape(-1,1)
    return np.mean(delta**2*( (1+((y_true-y_pred)/delta)**2)**0.5 -1))

def Modelevaluate(y_pred, y_true, yscale):

    yscaler = yscale
    y_true = yscaler.inverse_transform(y_true).reshape(-1, 1)
    y_pred = yscaler.inverse_transform(y_pred).reshape(-1, 1)

    mse = mean_squared_error(y_true,y_pred)
    R2  = r2_score(y_true,y_pred)
    mae = mean_absolute_error(y_true,y_pred)
    hub = huber(y_true,y_pred)

    return mse, np.sqrt(mse), R2, mae, hub


def plotpred(y_pred, y_true, yscale):

    yscaler = yscale
    y_true = yscaler.inverse_transform(y_true).reshape(-1, 1)
    y_pred = yscaler.inverse_transform(y_pred).reshape(-1, 1)

    plt.title('Inception')
    # plt.xticks(col)
    # plt.yticks(col)
    plt.scatter(y_true, y_pred)

    min_x = min(y_true.min(),y_pred.min())
    max_x = max(y_true.max(),y_pred.max())

    plt.plot([min_x, max_x], [min_x, max_x])

    plt.xlabel('True Value')
    plt.ylabel('Predicated Value')

    plt.show()


def plottransfer(y_true, pred, tranferprd, yscale,RMSE,R2):

    yscaler = yscale
    y_true = yscaler.inverse_transform(y_true).reshape(-1, 1)
    y_pred = yscaler.inverse_transform(pred).reshape(-1, 1)
    tranferprd = yscaler.inverse_transform(tranferprd).reshape(-1, 1)

    # plt.title('Inception')
    # plt.xticks(col)
    # plt.yticks(col)
    plt.scatter(y_true, y_pred, label='transfer before',color='r')
    plt.scatter(y_true, tranferprd, label='Transfer4',color='gold',marker='*')

    plt.legend(loc='best')

    plt.title('$RMSEP$={:.3f}    $R^2$={:.3f}'.format(RMSE,R2),fontstyle='italic',fontsize=13)
    # 'x= ' + str(x) + ', y = ' + str(y)

    min_x = min(y_true.min(),y_pred.min(),tranferprd.min())
    max_x = max(y_true.max(),y_pred.max(),tranferprd.max())

    plt.plot([min_x, max_x], [min_x, max_x])

    plt.xlabel('True Value')
    plt.ylabel('Predicated Value')

    plt.show()






