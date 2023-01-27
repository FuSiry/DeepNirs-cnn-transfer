import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from Data_Load import instrum2, MyDataset ,instrum1, corn, IDRC2016, KSIDRC2016
from InceptionResNet import IneptionRGSNet
from ConvNet import conv1
from earlystop import EarlyStopping
from datetime import datetime
from evaluate import Modelevaluate
from InceptionResNet import IneptionRGSNet, freeze_by_names
from sklearn.preprocessing import scale,MinMaxScaler,Normalizer,StandardScaler
from tensorboardX import SummaryWriter
import os

LR = 0.0001
BATCH_SIZE = 128
TBATCH_SIZE = 128

# path = './/model/IDCR2002/inception.pt'
path = './/model/IDRC2016/conv.pt'
result_path = './/Result//Test//A2//Incep.csv'
# store_path = './/model//IDCR2002//transfer//net.pt'
store_path = './/model//IDRC2016//transfer//net.pt'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# SumWriter = SummaryWriter(log_dir='runs/exp')


###定义是否需要标准化
def ZspPocessnew(sourcextrain, soureceytrain, X_train, X_test, y_train, y_test, need=True): #True:需要标准化，Flase：不需要标准化

    global standscale
    global yscaler

    if (need == True):
        standscale = StandardScaler()
        sourcetrain = standscale.fit_transform(sourcextrain)
        X_train_Nom = standscale.transform(X_train)
        X_test_Nom = standscale.transform(X_test)

        yscaler = StandardScaler()
        sourecetest = yscaler.fit_transform(soureceytrain)
        y_train = yscaler.transform(y_train)
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

def Test1(sourcextrain, sourceytrain, X_train, X_test, y_train, y_test, EPOCH):

    data_train, data_test = ZspPocessnew(sourcextrain, sourceytrain, X_train, X_test, y_train, y_test, need=True)
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=TBATCH_SIZE, shuffle=True)

    Model_Pretrained = conv1()  # 加载训练模型

    # model = nn.Sequential(*list(Model_Pretrained.children())).to(device)
    model = Model_Pretrained.to(device)

    # for m in model.parameters():
    #     if isinstance(m, nn.Conv2d):
    #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #         m.weight.data.normal_(0, math.sqrt(2. / n))
    #     elif isinstance(m, nn.BatchNorm2d):
    #         m.weight.data.fill_(1)
    #         m.bias.data.zero_()

    ModelStore = torch.load(path)  # 加载预训练模型参数
    ModelDict = Model_Pretrained.state_dict()  # 加载训练模型网络层数
    StateDict = {k: v for k, v in ModelStore.items() if k in ModelDict.keys()}  # 匹配预训练模型的网络参数和现有模型网络层之间的对应
    ModelDict.update(StateDict)  # 更新
    Model_Pretrained.load_state_dict(ModelDict)  # 现有模型载入参数


    criterion = nn.MSELoss().to(device)  # 损失函数为焦损函数，多用于类别不平衡的多分类问题
    # freeze_by_names(model, ('conv1', 'Inception1', 'Inception2', 'Inception3'))
    # freeze_by_names(model, ('conv1', 'conv2'))
    # model.conv1.weight.requires_grad = False
    # model.Inception1.weight.requires_grad = False
    # model.Inception2.weight.requires_grad = False
    # model.Inception3.weight.requires_grad = False
    #optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, momentum=0.9)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=50, path=store_path, verbose=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, verbose=1, eps=1e-06,
                                                           patience=20)
    print("Start Training!")  # 定义遍历数据集的次数
    # to track the training loss as the model trains
    best_loss = 2.9
    avg_train_losses = []
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
            train_mse.append(mse)
            train_rmse.append(rmse)
            train_r2.append(R2)
            train_mae.append(mae)
            train_hub.append(hub)
        avg_train_loss = np.mean(train_losses)
        # avg_train_losses.append(avg_train_loss)
        avgmse = np.mean(train_mse)
        avgrmse = np.mean(train_rmse)
        # SumWriter.add_scalar('RMSE',avgrmse,global_step=epoch)
        avgr2 = np.mean(train_r2)
        avgmae = np.mean(train_mae)
        avghub = np.mean(train_hub)


        # for name, param in model.named_parameters():
        #     SumWriter.add_histogram(name,param.data)



        with torch.no_grad():  # 无梯度
            with open(result_path, "w") as f1:
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
                    test_mse.append(mse)
                    test_rmse.append(rmse)
                    test_r2.append(R2)
                    test_mae.append(mae)
                    test_hub.append(hub)
                avgmse = np.mean(test_mse)
                avgrmse = np.mean(test_rmse)
                # SumWriter.add_scalar('TESTRMSE', avgrmse, global_step=epoch)
                avgr2   = np.mean(test_r2)
                # SumWriter.add_scalar('TESTR2', avgr2, global_step=epoch)
                avgmae = np.mean(test_mae)
                # SumWriter.add_scalar('TESTRMAE', avgmae, global_step=epoch)
                avghub = np.mean(test_hub)
                # SumWriter.add_scalar('TESTHUBE', avghub, global_step=epoch)
                # if avgrmse < best_loss:
                #     best_loss = avgrmse
                #     f3 = open(".//Result//Test//Transferbestloss.txt", "w")
                #     f3.write("EPOCH：{}, rmse:{}, R2:{}, mae:{}, hub:{}" .format(epoch + 1, best_loss, avgr2, avgmae, avghub))
                #     f3.close()
                scheduler.step(avgmse)

                # if epoch > 150:
                early_stopping(avgrmse, model)
                if early_stopping.early_stop:
                    print(f'Early stopping! Best validation loss: {early_stopping.get_best_score()}')
                    break

                print('Test:Epoch:{}, TRAIN:mse:{}, rmse:{}, R2:{}, mae:{}, hub:{}'.format((epoch + 1), (avgmse), (avgrmse),
                                                                                      (avgr2), (avgmae), (avghub)))
                print('lr:{}, avg_train_loss:{}'.format((optimizer.param_groups[0]['lr']), avg_train_loss))
                # f1.write("{},{},{},{},{},{}".format((epoch+1), (avgmse), (avgrmse), (avgr2), (avgmae), (avghub)))  # 写入数据
                # f1.write('\n')
                # print('EPOCH：{}, TEST:mse:{}, rmse:{}, R2:{}, mae:{}, hub:{}'.format((epoch+1),(avgmse), (avgrmse), (avgr2), (avgmae), (avghub)))
                # 将每次测试结果实时写入acc.txt文件中

def Test2(sourcextrain, sourceytrain, X_train, X_test, y_train, y_test, EPOCH):

    data_train, data_test = ZspPocessnew(sourcextrain, sourceytrain, X_train, X_test, y_train, y_test, need=True)
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=TBATCH_SIZE, shuffle=True)

    Model_Pretrained = IneptionRGSNet()  # 加载训练模型

    # model = nn.Sequential(*list(Model_Pretrained.children())).to(device)
    model = Model_Pretrained.to(device)

    # for m in model.parameters():
    #     if isinstance(m, nn.Conv2d):
    #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #         m.weight.data.normal_(0, math.sqrt(2. / n))
    #     elif isinstance(m, nn.BatchNorm2d):
    #         m.weight.data.fill_(1)
    #         m.bias.data.zero_()

    ModelStore = torch.load(path)  # 加载预训练模型参数
    ModelDict = Model_Pretrained.state_dict()  # 加载训练模型网络层数
    StateDict = {k: v for k, v in ModelStore.items() if k in ModelDict.keys()}  # 匹配预训练模型的网络参数和现有模型网络层之间的对应
    ModelDict.update(StateDict)  # 更新
    Model_Pretrained.load_state_dict(ModelDict)  # 现有模型载入参数


    criterion = nn.MSELoss().to(device)  # 损失函数为焦损函数，多用于类别不平衡的多分类问题
    # freeze_by_names(model, ('conv1', 'Inception1', 'Inception2', 'Inception3'))
    # model.conv1.weight.requires_grad = False
    # model.Inception1.weight.requires_grad = False
    # model.Inception2.weight.requires_grad = False
    # model.Inception3.weight.requires_grad = False
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, momentum=0.9)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=50, delta=1e-4, path=store_path, verbose=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, verbose=1, eps=1e-06,
                                                           patience=20)
    print("Start Training!")  # 定义遍历数据集的次数
    # to track the training loss as the model trains
    best_loss = 2.9
    avg_train_losses = []
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
            train_mse.append(mse)
            train_rmse.append(rmse)
            train_r2.append(R2)
            train_mae.append(mae)
            train_hub.append(hub)
        avg_train_loss = np.mean(train_losses)
        # avg_train_losses.append(avg_train_loss)
        avgmse = np.mean(train_mse)
        avgrmse = np.mean(train_rmse)
        # SumWriter.add_scalar('RMSE',avgrmse,global_step=epoch)
        avgr2 = np.mean(train_r2)
        avgmae = np.mean(train_mae)
        avghub = np.mean(train_hub)


        # for name, param in model.named_parameters():
        #     SumWriter.add_histogram(name,param.data)



        with torch.no_grad():  # 无梯度
            with open(result_path, "w") as f1:
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
                    test_mse.append(mse)
                    test_rmse.append(rmse)
                    test_r2.append(R2)
                    test_mae.append(mae)
                    test_hub.append(hub)
                avgmse = np.mean(test_mse)
                avgrmse = np.mean(test_rmse)
                # SumWriter.add_scalar('TESTRMSE', avgrmse, global_step=epoch)
                avgr2   = np.mean(test_r2)
                # SumWriter.add_scalar('TESTR2', avgr2, global_step=epoch)
                avgmae = np.mean(test_mae)
                # SumWriter.add_scalar('TESTRMAE', avgmae, global_step=epoch)
                avghub = np.mean(test_hub)
                # SumWriter.add_scalar('TESTHUBE', avghub, global_step=epoch)
                # if avgrmse < best_loss:
                #     best_loss = avgrmse
                #     f3 = open(".//Result//Test//Transferbestloss.txt", "w")
                #     f3.write("EPOCH：{}, rmse:{}, R2:{}, mae:{}, hub:{}" .format(epoch + 1, best_loss, avgr2, avgmae, avghub))
                #     f3.close()
                scheduler.step(avgmse)

                # if epoch > 150:
                early_stopping(avgrmse, model)
                if early_stopping.early_stop:
                    print(f'Early stopping! Best validation loss: {early_stopping.get_best_score()}')
                    break

                print('Test:Epoch:{}, TRAIN:mse:{}, rmse:{}, R2:{}, mae:{}, hub:{}'.format((epoch + 1), (avgmse), (avgrmse),
                                                                                      (avgr2), (avgmae), (avghub)))
                print('lr:{}, avg_train_loss:{}'.format((optimizer.param_groups[0]['lr']), avg_train_loss))
                # f1.write("{},{},{},{},{},{}".format((epoch+1), (avgmse), (avgrmse), (avgr2), (avgmae), (avghub)))  # 写入数据
                # f1.write('\n')
                # print('EPOCH：{}, TEST:mse:{}, rmse:{}, R2:{}, mae:{}, hub:{}'.format((epoch+1),(avgmse), (avgrmse), (avgr2), (avgmae), (avghub)))
                # 将每次测试结果实时写入acc.txt文件中

def tansferTest(sourcetrain, sourecetest, X_train, X_test, y_train, y_test, EPOCH):
    data_train, data_test = ZspPocessnew(sourcetrain, sourecetest, X_train, X_test, y_train, y_test, need=True)
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=BATCH_SIZE, shuffle=True)

    Model_Pretrained = IneptionRGSNet()  # 加载训练模型

    # model = nn.Sequential(*list(Model_Pretrained.children())).to(device)
    model = Model_Pretrained.to(device)

    ModelStore = torch.load(path)  # 加载预训练模型参数
    ModelDict = Model_Pretrained.state_dict()  # 加载训练模型网络层数
    StateDict = {k: v for k, v in ModelStore.items() if k in ModelDict.keys()}  # 匹配预训练模型的网络参数和现有模型网络层之间的对应
    ModelDict.update(StateDict)  # 更新
    Model_Pretrained.load_state_dict(ModelDict)  # 现有模型载入参数
    print(model)

    # criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵，多用于多分类问题
    # optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=0.03)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, verbose=1, eps=1e-06,
    #                                                        patience=20)

    for epoch in range(EPOCH):
        model.eval()  # 不训练
        best_loss = 3.0

        with open(result_path, "w") as f1:
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
                test_mse.append(mse)
                test_rmse.append(rmse)
                test_r2.append(R2)
                test_mae.append(mae)
                test_hub.append(hub)
            avgmse = np.mean(test_mse)
            avgrmse = np.mean(test_rmse)
            avgr2 = np.mean(test_r2)
            avgmae = np.mean(test_mae)
            avghub = np.mean(test_hub)
            if avgrmse < best_loss:
                best_loss = avgrmse
                f3 = open(".//Result//Test//Transferbestloss.txt", "a")
                f3.write(
                    "EPOCH：{}, rmse:{}, R2:{}, mae:{}, hub:{}".format(epoch + 1, best_loss, avgr2, avgmae, avghub))
                f3.write('\n')
                f3.close()
            f1.write(
                "{},{},{},{},{},{}".format((epoch + 1), (avgmse), (avgrmse), (avgr2), (avgmae), (avghub)))  # 写入数据
            f1.write('\n')
            print('EPOCH：{}, TEST:mse:{}, rmse:{}, R2:{}, mae:{}, hub:{}'.format((epoch + 1), (avgmse), (avgrmse),
                                                                                 (avgr2), (avgmae), (avghub)))
            # 将每次测试结果实时写入acc.txt文件中



if __name__ == "__main__":

    # X_train1, y_train1, X_test1, y_test1 = instrum1()
    # X_train2, y_train2, X_test2, y_test2 = instrum2()

    # X_train1, y_train1, X_test1, y_test1 = corn('m5')
    # X_train2, y_train2, X_test2, y_test2 = corn('mp5')

    X_train1, y_train1, X_test1, y_test1 = KSIDRC2016('A1', 200)

    X_train2, y_train2, X_test2, y_test2 = KSIDRC2016('A2', 30)


    # print('训练集规模：{}'.format(len(X_train[:, 0])))
    # print('测试集规模：{}'.format(len(X_test[:, 0])))
    # print('X_train：{}'.format(X_train.shape))
    # print('y_train：{}'.format(X_train.shape))


    #
    # print(model)
    # tansferTest(X_train, X_test, y_train, y_test,EPOCH=300)
    Test1(X_train1, y_train1, X_test2, X_train2, y_test2, y_train2, EPOCH=600)
    # Test2(X_train1, y_train1, X_test2, X_train2, y_test2, y_train2, EPOCH=200)