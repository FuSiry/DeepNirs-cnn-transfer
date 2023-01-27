import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from Data_Load import instrum2, MyDataset ,instrum1,  IDRC2016, KSIDRC2016
from ConvNet import conv1
from earlystop import EarlyStopping
from datetime import datetime
from evaluate import Modelevaluate
from InceptionResNet import IneptionRGSNet16, freeze_by_names, IneptionARGSNet, IneptionARGSNet16
from sklearn.preprocessing import scale,MinMaxScaler,Normalizer,StandardScaler
from evaluate import plottransfer
# from tensorboardX import SummaryWriter
import os
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)

###定义是否需要标准化
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
    else:
        X_train = X_train[:, np.newaxis, :]  # （483， 1， 2074）
        X_test = X_test[:, np.newaxis, :]
        data_train = MyDataset(X_train, y_train)
        ##使用loader加载测试数据
        data_test = MyDataset(X_test, y_test)
        return data_train, data_test


def TransferTrain(TransferType, X_train, X_test, y_train, y_test, EPOCH, base_path):
    pass



def ModelTransfertest(NetType, X_train, X_test, y_train, y_test, base_path):

    data_train, data_test = ZspPocessnew(X_train, X_test, y_train, y_test, need=True)

    test_loader = torch.utils.data.DataLoader(data_test, batch_size=TBATCH_SIZE, shuffle=False)

    # load the last checkpoint with the best model
    if NetType == 'Transfer1' or NetType == 'Transfer2':
        model = IneptionARGSNet16().to(device)
    elif NetType == 'Transfer3' :
        Model_Pretrained = IneptionARGSNet16()  # 加载训练模型
        model = nn.Sequential(*list(Model_Pretrained.children())[:-1],  # [b, 512, 1, 1]
                              Flatten(),  # [b, 512, 1, 1] => [b, 512]
                              nn.Linear(5000, 1),
                              ).to(device)
    elif NetType == 'Transfer4' :
        Model_Pretrained = IneptionARGSNet16()  # 加载训练模型
        model = nn.Sequential(*list(Model_Pretrained.children())[:-1],  # [b, 512, 1, 1]
                              Flatten(),  # [b, 512, 1, 1] => [b, 512]
                              nn.Linear(5000, 1),
                              ).to(device)
    # elif
    # store_path = base_path + NetType + '12.pt'
    # store_path = base_path + NetType + '.pt'
    store_path = base_path

    model.load_state_dict(torch.load(store_path, map_location='cpu'))

    # # initialize the early_stopping object
    print("Start Training!")  # 定义遍历数据集的次数
    # to track the training loss as the model trains

    model.eval()  # 不训练
    test_rmse = []
    test_r2 = []

    for i, data in enumerate(test_loader):
        inputs, labels = data  # 输入和标签都等于data
        inputs = Variable(inputs).type(torch.FloatTensor).to(device)  # batch x
        labels = Variable(labels).type(torch.FloatTensor).to(device)  # batch y
        outputs = model(inputs)  # 输出等于进入网络后的输入
        pred = outputs.detach().cpu().numpy()
        y_true = labels.detach().cpu().numpy()
        mse, rmse, R2, mae, hub = Modelevaluate(pred, y_true,yscaler)
        # plotpred(pred, y_true, yscaler)

        test_rmse.append(rmse)
        test_r2.append(R2)

    avgrmse = np.mean(test_rmse)
    avgr2   = np.mean(test_r2)

    print('TEST:rmse:{}, R2:{}'.format((avgrmse), (avgr2)))

    return pred, y_true,avgrmse,avgr2

def ModelTest(X_train, X_test, y_train, y_test):

    data_train, data_test = ZspPocessnew(X_train, X_test, y_train, y_test, need=True)

    test_loader = torch.utils.data.DataLoader(data_test, batch_size=TBATCH_SIZE, shuffle=False)

    # load the last checkpoint with the best model
    model = IneptionARGSNet16().to(device)

    store_path = './/model//IDRC2016//IncepIDRC20162.pt'

    model.load_state_dict(torch.load(store_path,  map_location='cpu'))

    # # initialize the early_stopping object
    print("Start Training!")  # 定义遍历数据集的次数
    # to track the training loss as the model trains

    model.eval()  # 不训练
    test_rmse = []
    test_r2 = []

    for i, data in enumerate(test_loader):
        inputs, labels = data  # 输入和标签都等于data
        inputs = Variable(inputs).type(torch.FloatTensor).to(device)  # batch x
        labels = Variable(labels).type(torch.FloatTensor).to(device)  # batch y
        outputs = model(inputs)  # 输出等于进入网络后的输入
        pred = outputs.detach().cpu().numpy()
        y_true = labels.detach().cpu().numpy()
        mse, rmse, R2, mae, hub = Modelevaluate(pred, y_true, yscaler)
        # plotpred(pred, y_true, yscaler)
        test_rmse.append(rmse)
        test_r2.append(R2)
    avgrmse = np.mean(test_rmse)
    avgr2 = np.mean(test_r2)

    print('TEST:rmse:{}, R2:{}'.format((avgrmse), (avgr2)))

    return pred, y_true

def ConvTransfer(TransferType,X_train, X_test, y_train, y_test, EPOCH, base_path):

    data_train, data_test = ZspPocessnew(X_train, X_test, y_train, y_test, need=True)
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=TBATCH_SIZE, shuffle=True)

    #第一种迁移：模型所有层迁移，不冻结任何网络层


if __name__ == "__main__":


    LR = 0.0001
    BATCH_SIZE = 75 #95
    TBATCH_SIZE = 220

    path = './/model/IDRC2016/IncepIDRC2016.pt'
    # path = './/model/IDRC2016/conv2.pt'

    # X_train1, y_train1, X_test1, y_test1 = KSIDRC2016('A1', 190)
    # X_train2, y_train2, X_test2, y_test2 = KSIDRC2016('A2', 70)


    X_train2, y_train2, X_test2, y_test2 = IDRC2016('B1', 0.8, randseed=123) #13

    # root = './/model//IDRC2016//transfer//30//'
    #root = './/model/IDRC2016/IncepIDRC2016.pt'
    root = './/model//IDRC2016//transfer//30//'

    X_train, y_train, X_test, y_test = X_train2, y_train2, X_test2, y_test2


    # convroot = './/model//IDRC2016//transfer//conv//'

    # ConvTransfer(TransferType='Transfer1', X_train=X_train, X_test=X_test,
    #               y_train= y_train, y_test=y_test, EPOCH=600,base_path=convroot)
    # ConvTransfer(TransferType='Transfer2', X_train=X_train, X_test=X_test,
    #              y_train=y_train, y_test=y_test, EPOCH=600, base_path=convroot)
    # ConvTransfer(TransferType='Transfer3', X_train=X_train, X_test=X_test,
    #              y_train=y_train, y_test=y_test, EPOCH=600, base_path=convroot)
    # ConvTransfer(TransferType='Transfer4', X_train=X_train, X_test=X_test,
    #              y_train=y_train, y_test=y_test, EPOCH=600, base_path=convroot)



    # start = time.time()
    # TransferTrain(TransferType='Transfer2', X_train=X_train, X_test=X_test,
    #               y_train= y_train, y_test=y_test, EPOCH=600,base_path=root)
    # end1 = time.time()
    # TransferTrain(TransferType='Transfer2', X_train=X_train, X_test=X_test,
    #               y_train= y_train, y_test=y_test, EPOCH=600,base_path=root)
    # end2 = time.time()
    # TransferTrain(TransferType='Transfer3', X_train=X_train, X_test=X_test,
    #               y_train= y_train, y_test=y_test, EPOCH=1000,base_path=root)
    # end3 = time.time()
    # TransferTrain(TransferType='Transfer4', X_train=X_train, X_test=X_test,
    #               y_train= y_train, y_test=y_test, EPOCH=1000,base_path=root)
    #
    #
    Tpred, Ty_true,RMSE,R2  = ModelTransfertest(NetType='Transfer1',X_train=X_train, X_test=X_test,
                  y_train= y_train, y_test=y_test ,base_path=root)
    # #
    pred, y_true= ModelTest(X_train=X_train, X_test=X_test,
                  y_train= y_train, y_test=y_test)
    #
    plottransfer(y_true=Ty_true, pred=pred, tranferprd=Tpred, yscale=yscaler,RMSE=RMSE,R2=R2)



    # Tpred, Ty_true,RMSE,R2  = ModelTransfertest(NetType='Transfer1',X_train=X_train, X_test=X_test,
    #               y_train= y_train, y_test=y_test ,base_path=root)

    # end4 = time.time()
    #
    # way1 = end1 - start
    # way2 = end2 - end1
    # way3 = end3 - end2
    # way4 = end4 - end3

    # StorePath = root + 'time.txt'
    # with open(StorePath,'w') as f1:
    #     f1.write('tf1 time:{},tf2 time:{},tf3 time:{},tf4 time:{}'.format(way1,way2,way3,way4))