import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号


train_file = pd.read_csv('.//Result//Train//Inception.csv')
test_file  = pd.read_csv('.//Result//Test//Inception.csv')


print(train_file)

df_train = train_file.groupby('epoch').mean()
df_test  = test_file.groupby('epoch').mean()

print(df_train)
print(df_test)

train_acc_list = []

length = len(df_train.index)
acc_list = df_train.loc[:,['acc']]
loss_list = df_train.loc[:,['loss']]




test_loss_list = df_test.loc[:,['loss']]
test_acc_list = df_test.loc[:,['acc']]
#
x_epoch = np.linspace(0, length, length)
plt.figure(500)
plt.plot(x_epoch, loss_list, 'r', label='train_loss')
plt.plot(x_epoch, test_loss_list, 'b', label='test_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss 曲线')
plt.legend(loc='best')
plt.show()

x_epoch = np.linspace(0, length, length)
plt.figure(500)
plt.plot(x_epoch, acc_list, 'r', label='train_acc')
plt.plot(x_epoch, test_acc_list, 'b', label='test_acc')
plt.xlabel('Epoch')
plt.ylabel('ACC')
plt.title('ACC 曲线')
plt.legend(loc='best')
plt.show()



train_file = pd.read_csv('.//Result//Train//InceptionRgs.csv')
test_file  = pd.read_csv('.//Result//Test//InceptionRgs.csv')


print(train_file)

df_train = train_file.groupby('epoch').mean()
df_test  = test_file.groupby('epoch').mean()

print(df_train)
print(df_test)

train_acc_list = []

length = len(df_train.index)
acc_list = df_train.loc[:,['acc']]


test_loss_list = df_test.loc[:,['loss']]
#
x_epoch = np.linspace(0, 300, 300)
plt.figure(500)
plt.plot(x_epoch, loss_list, 'r', label='train_loss')
plt.plot(x_epoch, test_loss_list, 'b', label='test_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss 曲线')
plt.legend(loc='best')
plt.show()

