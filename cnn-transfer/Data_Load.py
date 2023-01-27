"""
    Create on 2021-1-21
    Author：Pengyou Fu
    Describe：load drug data
"""

import numpy as np
import  pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
import  matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale,MinMaxScaler,Normalizer,StandardScaler
from KS import kennardstonealgorithm
import os


plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号


#139, 208,256,  415, 484,553
##自定义加载数据集
class MyDataset(Dataset):
    def __init__(self,specs,labels):
        self.specs = specs
        self.labels = labels

    def __getitem__(self, index):
        spec,target = self.specs[index],self.labels[index]
        return spec,target

    def __len__(self):
        return len(self.specs)

###定义是否需要标准化
def ZspPocess(X_train, X_test,y_train,y_test,need=True): #True:需要标准化，Flase：不需要标准化
    if (need == True):
        X_train_Nom = scale(X_train)
        X_test_Nom = scale(X_test)
        X_train_Nom = X_train_Nom[:, np.newaxis, :]
        X_test_Nom = X_test_Nom[:, np.newaxis, :]
        data_train = MyDataset(X_train_Nom, y_train)
        ##使用loader加载测试数据
        data_test = MyDataset(X_test_Nom, y_test)
        return data_train, data_test
    else:
        X_train = X_train[:, np.newaxis, :]  # （483， 1， 2074）
        X_test = X_test[:, np.newaxis, :]
        data_train = MyDataset(X_train, y_train)
        ##使用loader加载测试数据
        data_test = MyDataset(X_test, y_test)
        return data_train, data_test





#############################回归数据##################################
def instrum1():
    CDataPath1 = './/Data//shootout2002//Cdata1.csv'
    VDataPath1 = './/Data//shootout2002//Vdata1.csv'
    TDataPath1 = './/Data//shootout2002//Tdata1.csv'
    Cdata1 = np.loadtxt(open(CDataPath1, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
    Vdata1 = np.loadtxt(open(VDataPath1, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
    Tdata1 = np.loadtxt(open(TDataPath1, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)

    Tedata = np.concatenate((Cdata1,Vdata1))

    idx1 = [4, 8, 10, 144, 266, 293, 294, 312, 340, 341, 343] #train 里面使用
    idx2 = [18, 121, 125, 126, 149] #test 里面使用

    Sdata = np.delete(Tdata1,idx1,axis=0)
    STdata  = np.delete(Tedata,idx2,axis=0)

    XTrain = Sdata[:,0:530]
    y_train = Sdata[:,-1]
    XTest = STdata[:,0:530]
    y_test = STdata[:,-1]

    # return XTrain, y_train, XTest, y_test
    return pd.DataFrame(XTrain),pd.DataFrame(y_train),pd.DataFrame(XTest),pd.DataFrame(y_test)


def instrum2():
    CDataPath1 = './/Data//shootout2002//Cdata2.csv'
    VDataPath1 = './/Data//shootout2002//Vdata2.csv'
    TDataPath1 = './/Data//shootout2002//Tdata2.csv'
    Cdata1 = np.loadtxt(open(CDataPath1, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
    Vdata1 = np.loadtxt(open(VDataPath1, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
    Tdata1 = np.loadtxt(open(TDataPath1, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)

    Tedata = np.concatenate((Cdata1, Vdata1))

    idx1 = [4, 8, 10, 144, 266, 293, 294, 312, 340, 341, 343]  # train 里面使用
    idx2 = [18, 121, 125, 126, 149]  # test 里面使用

    Sdata = np.delete(Tdata1, idx1, axis=0)
    STdata = np.delete(Tedata, idx2, axis=0)

    XTrain = Sdata[:, 0:530]
    y_train = Sdata[:, -1]
    XTest = STdata[:, 0:530]
    y_test = STdata[:, -1]

    return pd.DataFrame(XTrain), pd.DataFrame(y_train), pd.DataFrame(XTest), pd.DataFrame(y_test)



def KSIDRC2016(Dtpye,Nums):

    folder = './Data/IDRC2016'
    # cal_a_1 = pd.read_excel(os.path.join(folder,'Cal_ManufacturerA.xls'),sheet_name=0)
    # cal_a_2 = pd.read_excel(os.path.join(folder,'Cal_ManufacturerA.xls'),sheet_name=1)
    # cal_a_3 = pd.read_excel(os.path.join(folder,'Cal_ManufacturerA.xls'),sheet_name=2)
    # test_a = pd.read_excel(os.path.join(folder,'Test_ManufacturerA.xls'),sheet_name=0)
    # val_a = pd.read_excel(os.path.join(folder,'Val_ManufacturerA.xls'),sheet_name=0)
    #
    # cal_b_1 = pd.read_excel(os.path.join(folder,'Cal_ManufacturerB.xls'),sheet_name=0)
    # cal_b_2 = pd.read_excel(os.path.join(folder,'Cal_ManufacturerB.xls'),sheet_name=1)
    # cal_b_3 = pd.read_excel(os.path.join(folder,'Cal_ManufacturerB.xls'),sheet_name=2)
    # test_b = pd.read_excel(os.path.join(folder,'Test_ManufacturerB.xls'),sheet_name=0)
    # val_b = pd.read_excel(os.path.join(folder,'Val_ManufacturerB.xls'),sheet_name=0)

    # dataA = np.concatenate(cal_a_1[1:,2:],cal_a_2[1:,2:],cal_a_3[1:,2:],test_a[1:,2:])

    # cal_a_1 = np.loadtxt(os.path.join(folder,'SG2DA1.csv'), dtype=np.float64, delimiter=',', skiprows=0)
    # cal_a_2 = np.loadtxt(os.path.join(folder,'SG2DA2.csv'), dtype=np.float64, delimiter=',', skiprows=0)

    cal_a_1 = pd.read_csv(os.path.join(folder,'Cal_ManufacturerxA1.csv'))
    cal_a_2 = pd.read_csv(os.path.join(folder,'Cal_ManufacturerxA2.csv'))
    cal_a_3 = pd.read_csv(os.path.join(folder,'Cal_ManufacturerxA3.csv'))
    test_a = pd.read_csv(os.path.join(folder,'Test_ManufacturerA.csv'))
    val_a = pd.read_csv(os.path.join(folder,'Val_ManufacturerA.csv'))


    cal_a_1 = np.array(cal_a_1.iloc[0:,3:])
    cal_a_2 = np.array(cal_a_2.iloc[0:,3:])

    # cal_a_3 = np.array(cal_a_3.iloc[0:,2:])
    # val_a = np.array(val_a.iloc[0:, 2:])
    # test_a = np.array(test_a.iloc[0:, 2:])

    if Dtpye == 'A1' :

        global train_index
        global test_index

        # data_x = np.array(cal_a_1[:,0:500])
        data_x = np.array(cal_a_1[:,0:])

        # np.savetxt('A1.csv', data_x, delimiter=',')
        label = cal_a_3.iloc[0:, 2]
        data_y = np.array(label)

        badindx = [187]

        data_x = np.delete(data_x, badindx, axis=0)
        data_y = np.delete(data_y, badindx, axis=0)

        selected_sample_numbers, remaining_sample_numbers = kennardstonealgorithm(data_x, Nums)
        train_index = selected_sample_numbers
        test_index = remaining_sample_numbers

        train_list = ''
        test_list = ''

        # train_numbers_path = os.path.join(folder,'A1TrainIndex.csv')
        # test_numbers_path = os.path.join(folder, 'A1TestIndex.csv')
        #
        # with open(train_numbers_path, mode='w') as f1:
        #     for index in selected_sample_numbers:
        #         train_list += str(index)
        #         train_list += ','
        #     f1.write(train_list)
        #
        # with open(test_numbers_path, mode='w') as f2:
        #     for idx in  remaining_sample_numbers:
        #         test_list += str(idx)
        #         test_list += ','
        #     f2.write(test_list)

        X_train = np.delete(data_x, test_index, axis=0)
        X_test = np.delete(data_x, train_index, axis=0)
        y_train = np.delete(data_y, test_index, axis=0)
        y_test = np.delete(data_y, train_index, axis=0)

    elif Dtpye == 'A2':
        data_x = np.array(cal_a_2[:,0:]) #[:,0:500]

        # np.savetxt('A2.csv', data_x, delimiter=',')

        label = cal_a_3.iloc[0:, 2]
        data_y = np.array(label)

        badindx = [187]

        data_x = np.delete(data_x, badindx, axis=0)
        data_y = np.delete(data_y, badindx, axis=0)

        plt.figure(50)
        x_col = np.linspace(0,741,741)  # 数组逆序
        y_col = np.transpose(data_x)
        plt.plot(x_col, y_col)
        plt.xlabel("Wavenumber(nm)")
        plt.ylabel("Absorbance")
        plt.title("The spectrum of the source dataset", fontweight="semibold", fontsize='x-large')
        plt.show()

        selected_sample_numbers, remaining_sample_numbers = kennardstonealgorithm(data_x, Nums)
        train_index = selected_sample_numbers
        test_index = remaining_sample_numbers

        # print(train_index)

        X_train = np.delete(data_x, test_index, axis=0)
        X_test = np.delete(data_x, train_index, axis=0)
        y_train = np.delete(data_y, test_index, axis=0)
        y_test = np.delete(data_y, train_index, axis=0)



    elif Dtpye == 'A3':
        data_x = np.array(cal_a_3.iloc[0:,3:])
        label = cal_a_3.iloc[0:, 2]
        data_y = np.array(label)

        selected_sample_numbers, remaining_sample_numbers = kennardstonealgorithm(data_x, Nums)
        train_index = selected_sample_numbers
        test_index = remaining_sample_numbers

        X_train = np.delete(data_x, test_index, axis=0)
        X_test = np.delete(data_x, train_index, axis=0)
        y_train = np.delete(data_y, test_index, axis=0)
        y_test = np.delete(data_y, train_index, axis=0)

    elif Dtpye == 'A5':
        data_x = np.array(test_a.iloc[0:,3:])
        label = test_a.iloc[0:, 2]
        data_y = np.array(label)

        selected_sample_numbers, remaining_sample_numbers = kennardstonealgorithm(data_x, Nums)
        train_index = selected_sample_numbers
        test_index = remaining_sample_numbers

        X_train = np.delete(data_x, test_index, axis=0)
        X_test = np.delete(data_x, train_index, axis=0)
        y_train = np.delete(data_y, test_index, axis=0)
        y_test = np.delete(data_y, train_index, axis=0)

    return pd.DataFrame(X_train), pd.DataFrame(y_train), pd.DataFrame(X_test), pd.DataFrame(y_test)

def IDRC2016(Dtpye,size,randseed):

    folder = './Data/IDRC2016'
    # cal_a_1 = pd.read_excel(os.path.join(folder,'Cal_ManufacturerA.xls'),sheet_name=0)
    # cal_a_2 = pd.read_excel(os.path.join(folder,'Cal_ManufacturerA.xls'),sheet_name=1)
    # cal_a_3 = pd.read_excel(os.path.join(folder,'Cal_ManufacturerA.xls'),sheet_name=2)
    # test_a = pd.read_excel(os.path.join(folder,'Test_ManufacturerA.xls'),sheet_name=0)
    # val_a = pd.read_excel(os.path.join(folder,'Val_ManufacturerA.xls'),sheet_name=0)

    # cal_a_1 = pd.read_csv(os.path.join(folder,'SG2DA1.csv'))
    # cal_a_2 = pd.read_csv(os.path.join(folder,'SG2DA2.csv'))

    # cal_a_1 = np.loadtxt(os.path.join(folder,'MSCA1.csv'), dtype=np.float64, delimiter=',', skiprows=0)
    # cal_a_2 = np.loadtxt(os.path.join(folder,'MSCA2.csv'), dtype=np.float64, delimiter=',', skiprows=0)


    cal_a_1 = pd.read_csv(os.path.join(folder,'Cal_ManufacturerxA1.csv'))
    cal_a_2 = pd.read_csv(os.path.join(folder,'Cal_ManufacturerxA2.csv'))
    cal_a_3 = pd.read_csv(os.path.join(folder,'Cal_ManufacturerxA3.csv'))
    test_a = pd.read_csv(os.path.join(folder,'Test_ManufacturerA.csv'))

    cal_b_1 = pd.read_csv(os.path.join(folder, 'Cal_ManufacturerB1.csv'))
    cal_b_2 = pd.read_csv(os.path.join(folder, 'Cal_ManufacturerB2.csv'))
    cal_b_3 = pd.read_csv(os.path.join(folder, 'Cal_ManufacturerB3.csv'))


    # cal_b_1 = pd.read_excel(os.path.join(folder,'Cal_ManufacturerB.xls'),sheet_name=0)
    # cal_b_2 = pd.read_excel(os.path.join(folder,'Cal_ManufacturerB.xls'),sheet_name=1)
    # cal_b_3 = pd.read_excel(os.path.join(folder,'Cal_ManufacturerB.xls'),sheet_name=2)
    # test_b = pd.read_excel(os.path.join(folder,'Test_ManufacturerB.xls'),sheet_name=0)
    # val_b = pd.read_excel(os.path.join(folder,'Val_ManufacturerB.xls'),sheet_name=0)

    # dataA = np.concatenate(cal_a_1[1:,2:],cal_a_2[1:,2:],cal_a_3[1:,2:],test_a[1:,2:])


    cal_a_1 = np.array(cal_a_1.iloc[0:,3:])
    cal_a_2 = np.array(cal_a_2.iloc[0:,3:])
    cal_b_1 = np.array(cal_b_1.iloc[0:,3:])
    cal_b_2 = np.array(cal_b_2.iloc[0:,3:])


    # cal_a_3 = np.array(cal_a_3.iloc[0:,2:])
    # val_a = np.array(val_a.iloc[0:, 2:])
    # test_a = np.array(test_a.iloc[0:, 2:])

    if Dtpye == 'A1' :
        data_x = np.array(cal_a_1[:,])

        label = cal_a_3.iloc[0:, 2]
        data_y = np.array(label)

        badindx = [187]

        data_x = np.delete(data_x, badindx, axis=0)
        data_y = np.delete(data_y, badindx, axis=0)

        X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=size,random_state=randseed)

    elif Dtpye == 'A2':
        data_x = np.array(cal_a_2[:,:])
        label = cal_a_3.iloc[0:, 2]
        data_y = np.array(label)

        badindx = [187]

        data_x = np.delete(data_x, badindx, axis=0)
        data_y = np.delete(data_y, badindx, axis=0)


        X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=size, random_state=randseed)

    elif Dtpye == 'A3':
        data_x = np.array(cal_a_3.iloc[0:,3:])
        label = cal_a_3.iloc[0:, 2]
        data_y = np.array(label)

        X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=size, random_state=randseed)

    elif Dtpye == 'A5':
        data_x = np.array(test_a.iloc[0:,3:])
        label = test_a.iloc[0:, 2]
        data_y = np.array(label)

        X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=size, random_state=randseed)

    elif Dtpye == 'B1':

        data_x = np.array(cal_b_1[:,:])
        label = cal_b_3.iloc[0:, 2]
        data_y = np.array(label)

        badindx = [187]

        data_x = np.delete(data_x, badindx, axis=0)
        data_y = np.delete(data_y, badindx, axis=0)

        X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=size, random_state=randseed)

    elif Dtpye == 'B2':

        data_x = np.array(cal_b_2[:, :])
        label = cal_b_3.iloc[0:, 2]
        data_y = np.array(label)

        badindx = [187]

        data_x = np.delete(data_x, badindx, axis=0)
        data_y = np.delete(data_y, badindx, axis=0)

        X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=size, random_state=randseed)


    return pd.DataFrame(X_train), pd.DataFrame(y_train), pd.DataFrame(X_test), pd.DataFrame(y_test)




if __name__ == '__main__':

    X_train, y_train, X_test, y_test = IDRC2016("B2", 0.2, 123)
    X_train , X_test = np.array(X_train), np.array(X_test)
    data = np.vstack((X_train, X_test))
    plt.figure(50)
    x_col = np.linspace(0, 1060, 1060)  # 数组逆序
    y_col = np.transpose(data)
    plt.plot(x_col, y_col)
    plt.xlabel("Wavenumber(nm)")
    plt.ylabel("Absorbance")
    plt.title("The spectrum of the source dataset", fontweight="semibold", fontsize='x-large')
    plt.show()


