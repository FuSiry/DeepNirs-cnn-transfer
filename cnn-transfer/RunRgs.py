"""
    Create on 2021-1-21
    Author：Pengyou Fu
    Describe：this for train NIRS with use 1-D Resnet model to transfer
"""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset
import torchvision
import torch.nn.functional as F
from sklearn.preprocessing import scale,MinMaxScaler,Normalizer,StandardScaler
import torch.optim as optim
from Data_Load import instrum1, MyDataset, instrum2,IDRC2016, KSIDRC2016
from InceptionResNet import IneptionRGSNet, IneptionARGSNet, incep1, IneptionRGSNet16, MSCnet, IneptionARGSNet16
from ConvNet import conv1
from earlystop import EarlyStopping
import os
from datetime import datetime
from evaluate import Modelevaluate,plotpred
import matplotlib.pyplot  as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error

LR = 0.001
BATCH_SIZE = 128#16
TBATCH_SIZE = 240


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# result_path = './/Result//test//IcdrInceptionRgs123.csv'
# store_path = 'model/IDRC2016/conv1.pt'
# store_path = 'model/'


###定义是否需要标准化
def ZspPocessnew(X_train, X_test, y_train, y_test, need=True): #True:需要标准化，Flase：不需要标准化

    global standscale
    global yscaler

    if (need == True):
        standscale = StandardScaler()
        X_train_Nom = standscale.fit_transform(X_train)
        X_test_Nom = standscale.transform(X_test)

        yscaler = StandardScaler()
        # yscaler = MinMaxScaler()
        y_train = yscaler.fit_transform(y_train)
        y_test = yscaler.transform(y_test)

        X_train_Nom = X_train_Nom[:, np.newaxis, :]
        X_test_Nom = X_test_Nom[:, np.newaxis, :]

        ##使用loader加载测试数据
        data_train = MyDataset(X_train_Nom, y_train)
        data_test = MyDataset(X_test_Nom, y_test)
        return data_train, data_test
    elif((need == False)):
        yscaler = StandardScaler()
        # yscaler = MinMaxScaler()

        X_train, X_test = np.array(X_train), np.array(X_test)
        X_train_new = X_train[:, np.newaxis, :]  #
        X_test_new = X_test[:, np.newaxis, :]


        y_train = yscaler.fit_transform(y_train)
        y_test = yscaler.transform(y_test)


        data_train = MyDataset(X_train_new, y_train)
        ##使用loader加载测试数据
        data_test = MyDataset(X_test_new, y_test)

        return data_train, data_test

def ZPocess(X_train, X_test, y_train, y_test): #True:需要标准化，Flase：不需要标准化

    global yscaler

    yscaler = StandardScaler()
    # yscaler = MinMaxScaler()
    y_train = yscaler.fit_transform(y_train)
    y_test = yscaler.transform(y_test)

    X_train = scale(X_train)
    X_test = scale(X_test)
    X_train_Nom = X_train[:, np.newaxis, :]
    X_test_Nom = X_test[:, np.newaxis, :]

    ##使用loader加载测试数据
    data_train = MyDataset(X_train_Nom, y_train)
    data_test = MyDataset(X_test_Nom, y_test)
    return data_train, data_test





def ModelTrain(NetType, X_train, X_test, y_train, y_test, EPOCH, base_path):


    data_train, data_test = ZspPocessnew(X_train, X_test, y_train, y_test, need=False)
    # data_train, data_test = ZPocess(X_train, X_test, y_train, y_test)

    train_loader = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=TBATCH_SIZE, shuffle=True)

    if NetType == 'conv':
        model = conv1().to(device)
    elif NetType == 'Incepthion':
        model = IneptionRGSNet16().to(device)
    elif NetType == 'IneptionARGSNet':
        model = IneptionARGSNet().to(device)
    elif NetType == 'MSCnet':
        model = MSCnet().to(device)
    elif NetType == 'IncepIDRC2016' :
        model = IneptionARGSNet16().to(device)


    store_path = base_path + NetType + '.pt'
    print(store_path)
    criterion = nn.MSELoss().to(device)  # 损失函数为焦损函数，多用于类别不平衡的多分类问题
    optimizer = optim.Adam(model.parameters(), lr=LR)#,  weight_decay=0.001)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
    # # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=60 , path=store_path, verbose=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, verbose=1, eps=1e-06,
                                                           patience=20)
    print("Start Training!")  # 定义遍历数据集的次数
    # to track the training loss as the model trains
    for epoch in range(EPOCH):
        train_losses = []
        model.train()  # 不训练
        train_mse = []
        train_rmse = []
        train_r2 = []
        train_mae = []
        train_hub = []
        for i, data in enumerate(train_loader):  # gives batch data, normalize x when iterate train_loader
            inputs, labels = data  # 输入和标签都等于data
            inputs = Variable(inputs).type(torch.FloatTensor).to(device)  # batch x
            labels = Variable(labels).type(torch.FloatTensor).to(device)  # batch y
            output = model(inputs)  # cnn output
            loss = criterion(output, labels)  # MSE
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            pred = output.detach().cpu().numpy()
            y_true = labels.detach().cpu().numpy()
            train_losses.append(loss.item())
            mse, rmse, R2, mae, hub = Modelevaluate(pred, y_true, yscaler)
            # plotpred(pred, y_true, yscaler)
            train_mse.append(mse)
            train_rmse.append(rmse)
            train_r2.append(R2)
            train_mae.append(mae)
            train_hub.append(hub)
        avg_train_loss = np.mean(train_losses)
        avgmse = np.mean(train_mse)
        avgrmse = np.mean(train_rmse)
        avgr2 = np.mean(train_r2)
        avgmae = np.mean(train_mae)
        avghub = np.mean(train_hub)
        print('Epoch:{}, TRAIN:mse:{}, rmse:{}, R2:{}, mae:{}, hub:{}'.format((epoch+1),(avgmse),(avgrmse),(avgr2),(avgmae),(avghub)))
        print('lr:{}, avg_train_loss:{}'.format((optimizer.param_groups[0]['lr']), avg_train_loss))

        with torch.no_grad():  # 无梯度
            model.eval()  # 不训练
            test_mse = []
            test_rmse = []
            test_r2 = []
            test_mae = []
            test_hub = []
            for i, data in enumerate(test_loader):
                inputs, labels = data  # 输入和标签都等于data
                inputs = Variable(inputs).type(torch.FloatTensor).to(device)  # batch x
                labels = Variable(labels).type(torch.FloatTensor).to(device)  # batch y
                outputs = model(inputs)  # 输出等于进入网络后的输入
                pred = outputs.detach().cpu().numpy()
                y_true = labels.detach().cpu().numpy()
                mse, rmse, R2, mae, hub = Modelevaluate(pred, y_true, yscaler)
                # plotpred(pred, y_true, yscaler)
                test_mse.append(mse)
                test_rmse.append(rmse)
                test_r2.append(R2)
                test_mae.append(mae)
                test_hub.append(hub)
            avgmse = np.mean(test_mse)
            avgrmse = np.mean(test_rmse)
            avgr2   = np.mean(test_r2)
            avgmae = np.mean(test_mae)
            avghub = np.mean(test_hub)
            print('EPOCH：{}, TEST:mse:{}, rmse:{}, R2:{}, mae:{}, hub:{}'.format((epoch+1),(avgmse), (avgrmse), (avgr2), (avgmae), (avghub)))
            # 将每次测试结果实时写入acc.txt文件中
            scheduler.step(avgmse)

            # if epoch > 150:
            early_stopping(avgrmse, model)
            if early_stopping.early_stop:
                print(f'Early stopping! Best validation loss: {early_stopping.get_best_score()}')
                break



def ModelTest(NetType, X_train, X_test, y_train, y_test, base_path):

    data_train, data_test = ZspPocessnew(X_train, X_test, y_train, y_test, need=False)
    # data_train, data_test = ZPocess(X_train, X_test, y_train, y_test)

    test_loader = torch.utils.data.DataLoader(data_test, batch_size=TBATCH_SIZE, shuffle=True)

    # load the last checkpoint with the best model
    if NetType == 'conv':
        model = conv1().to(device)
    elif NetType == 'Incepthion':
        model = IneptionRGSNet16().to(device)
    elif NetType == 'IneptionARGSNet':
        model = IneptionARGSNet().to(device)
    elif NetType == 'MSCnet':
        model = MSCnet().to(device)
    elif NetType == 'IncepIDRC2016' :
        model = IneptionARGSNet16().to(device)


    store_path = base_path + NetType + '.pt'

    model.load_state_dict(torch.load(store_path))

    # # initialize the early_stopping object
    print("Start Training!")  # 定义遍历数据集的次数
    # to track the training loss as the model trains

    model.eval()  # 不训练
    test_mse = []
    test_rmse = []
    test_r2 = []
    test_mae = []
    test_hub = []

    for i, data in enumerate(test_loader):
        inputs, labels = data  # 输入和标签都等于data
        inputs = Variable(inputs).type(torch.FloatTensor).to(device)  # batch x
        labels = Variable(labels).type(torch.FloatTensor).to(device)  # batch y
        outputs = model(inputs)  # 输出等于进入网络后的输入
        pred = outputs.detach().cpu().numpy()
        y_true = labels.detach().cpu().numpy()
        mse, rmse, R2, mae, hub = Modelevaluate(pred, y_true,yscaler)
        plotpred(pred, y_true, yscaler)
        test_mse.append(mse)
        test_rmse.append(rmse)
        test_r2.append(R2)
        test_mae.append(mae)
        test_hub.append(hub)
    avgmse = np.mean(test_mse)
    avgrmse = np.mean(test_rmse)
    avgr2   = np.mean(test_r2)
    avgmae = np.mean(test_mae)
    avghub = np.mean(test_hub)

    print('TEST:mse:{}, rmse:{}, R2:{}, mae:{}, hub:{}, (MAE+SEP)/2:{}'.format((avgmse), (avgrmse), (avgr2), (avgmae), (avghub), ((avgrmse+avgmae)/2)))

def Dataset(set,Netype,IsTrain = False):
    if set == 'IDRC2002':
        #载入数据
        X_train1, y_train1, X_test1, y_test1 = instrum1()
        # x_train_msc = MSC(X_train1)
        # x_tESTmsc = MSC(X_test1)

        X_train2, y_train2, X_test2, y_test2 = instrum2()

        base_path = './/model//IDCR2002//'

    elif set == 'IDRC2016' :
        # 载入数据
        X_train1, y_train1, X_test1, y_test1 = KSIDRC2016('A1', 198)
        X_train2, y_train2, X_test2, y_test2 = KSIDRC2016('A2', 30)
        # X_train1, y_train1, X_test1, y_test1 = IDRC2016('A1', 0.2, randseed=123)
        # X_train2, y_train2, X_test2, y_test2 = IDRC2016('A2', 0.86, randseed=123)
        base_path = './/model//IDRC2016//'


    else :
        print('errors')

    # 模型训练
    if IsTrain:
        ModelTrain(Netype,X_train1, X_test1, y_train1, y_test1, EPOCH=600, base_path=base_path)
    #预测改
    ModelTest(Netype,X_train1, X_test1, y_train1, y_test1, base_path=base_path)
    # #直接预测
    # print('A1')
    # ModelTest(Netype,X_train1, X_train2, y_train1, y_train2, base_path=base_path)
    # print('A2')
    # ModelTest(Netype, X_train2, X_test2, y_train2, y_test2, base_path=base_path)






if __name__ == "__main__":


    Dataset('IDRC2016', Netype='IncepIDRC2016', IsTrain = False)
