import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Iterable



def set_freeze_by_names(model, layer_names, freeze=True):
    if not isinstance(layer_names, Iterable):
        layer_names = [layer_names]
    for name, child in model.named_children():
        if name not in layer_names:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze


def freeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, True)


def unfreeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, False)


def set_freeze_by_idxs(model, idxs, freeze=True):
    if not isinstance(idxs, Iterable):
        idxs = [idxs]
    num_child = len(list(model.children()))
    idxs = tuple(map(lambda idx: num_child + idx if idx < 0 else idx, idxs))
    for idx, child in enumerate(model.children()):
        if idx not in idxs:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze


def freeze_by_idxs(model, idxs):
    set_freeze_by_idxs(model, idxs, True)


def unfreeze_by_idxs(model, idxs):
    set_freeze_by_idxs(model, idxs, False)


class InceptionA(nn.Module):
    def __init__(self,in_c,c1,c2,c3,c4,out_C):
        super(InceptionA,self).__init__()
        self.p1 = nn.Sequential(
            nn.Conv1d(in_c,c1,kernel_size=5,padding=2),
            nn.BatchNorm1d(c1),
            nn.ReLU()
        )
        self.p2 = nn.Sequential(
            nn.Conv1d(in_c,c2,kernel_size=9,padding=4),
            nn.BatchNorm1d(c2),
            nn.ReLU(),
        )
        self.p3 = nn.Sequential(
            nn.Conv1d(in_c, c3, kernel_size=15,padding=7),
            nn.BatchNorm1d(c3),
            nn.ReLU(),
        )
        self.p4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3,stride=1,padding=1),
            nn.Conv1d(in_c,c4,kernel_size=3,padding=1),
            nn.ReLU()
        )
        self.conv_linear = nn.Conv1d((c1+c2+c3+c4), out_C, 1, 1, 0, bias=True)
        self.short_cut = nn.Sequential()
        if in_c != out_C:
            self.short_cut = nn.Sequential(
                nn.Conv1d(in_c, out_C, 1, 1, 0, bias=False),
                # nn.BatchNorm1d(out_C),
                # nn.ReLU()
            )
    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)
        p4 = self.p4(x)
        out =  torch.cat((p1,p2,p3,p4),dim=1)
        # out = self.conv_linear(out)
        out += self.short_cut(x)
        return F.relu(out)

class Inception(nn.Module):
    def __init__(self,in_c,c1,c2,c3,c4,out_C):
        super(Inception,self).__init__()
        self.p1 = nn.Sequential(
            nn.Conv1d(in_c,c1,kernel_size=1),
            nn.ReLU()
        )
        self.p2 = nn.Sequential(
            nn.Conv1d(in_c,c2[0],kernel_size=1,padding=0),
            nn.ReLU(),
            nn.Conv1d(c2[0], c2[1], kernel_size=9,padding=4),
            nn.ReLU()
        )
        self.p3 = nn.Sequential(
            nn.Conv1d(in_c, c3[0], kernel_size=1,padding=0),
            nn.ReLU(),
            nn.Conv1d(c3[0], c3[1], kernel_size=15,padding=7),
            nn.ReLU()
        )
        self.p4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3,stride=1,padding=1),
            nn.Conv1d(in_c,c4,kernel_size=1),
            nn.ReLU()
        )
        self.conv_linear = nn.Conv1d((c1+c2[1]+c3[1]+c4), out_C, 1, 1, 0, bias=True)
        self.short_cut = nn.Sequential()
        if in_c != out_C:
            self.short_cut = nn.Sequential(
                nn.Conv1d(in_c, out_C, 1, 1, 0, bias=False),
                # nn.BatchNorm1d(out_C),
                # nn.ReLU()
            )
    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)
        p4 = self.p4(x)
        out =  torch.cat((p1,p2,p3,p4),dim=1)
        # out = self.conv_linear(out)
        out += self.short_cut(x)
        return F.relu(out)

class Inception16(nn.Module):
    def __init__(self,in_c,c1,c2,c3,c4,out_C):
        super(Inception16,self).__init__()
        self.p1 = nn.Sequential(
            nn.Conv1d(in_c,c1,kernel_size=1),
            nn.BatchNorm1d(c1),
            nn.ReLU()
        )
        self.p2 = nn.Sequential(
            nn.Conv1d(in_c,c2[0],kernel_size=1,padding=0),
            # nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(c2[0], c2[1], kernel_size=11,padding=5),
            nn.BatchNorm1d(c2[1]),
            nn.ReLU()
        )
        self.p3 = nn.Sequential(
            nn.Conv1d(in_c, c3[0], kernel_size=1,padding=0),
            # nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(c3[0], c3[1], kernel_size=17,padding=8),
            nn.BatchNorm1d(c3[1]),
            nn.ReLU()
        )
        self.p4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3,stride=1,padding=1),
            nn.Conv1d(in_c,c4,kernel_size=1),
            nn.BatchNorm1d(c4),
            nn.ReLU()
        )
        self.conv_linear = nn.Conv1d((c1+c2[1]+c3[1]+c4), out_C, 1, 1, 0, bias=True)
        self.short_cut = nn.Sequential()
        if in_c != out_C:
            self.short_cut = nn.Sequential(
                nn.Conv1d(in_c, out_C, 1, 1, 0, bias=False),
                nn.BatchNorm1d(out_C),
                # nn.ReLU()
            )
    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)
        p4 = self.p4(x)
        out =  torch.cat((p1,p2,p3,p4),dim=1)
        # out = self.conv_linear(out)
        out += self.short_cut(x)
        return F.relu(out)




class IneptionRGSNet(nn.Module):
    def __init__(self):
        super(IneptionRGSNet,self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=17, padding=0) #改之前17
        self.Inception1 = Inception(in_c=32, c1=20, c2=(10, 20), c3=(10, 20), c4=20, out_C=80)
        self.Inception2 = Inception(in_c=80, c1=25, c2=(16, 25), c3=(16, 25), c4=25, out_C=100)
        # self.Inception3 = Inception(in_c=130, c1=32, c2=(25, 32), c3=(25, 32), c4=32, out_C=128)
        self.mp = nn.MaxPool1d(3)
        self.AvgMaxPool = nn.AdaptiveMaxPool1d(70)
        self.fc1 = nn.Linear(7000, 1) #8960 ,7000, 5600
        # self.fc2 = nn.Linear(1000, 130)
        # self.fc3 = nn.Linear(130, 1)

    def forward(self,out):
      out = F.relu(self.conv1(out))
      # out = self.mp(F.relu(self.conv1(out)))
      # out = self.mp(F.relu(self.conv1(out)))
      out = self.Inception1(out)
      out = self.Inception2(out)
      # out = self.Inception3(out)
      # out = self.conv3(out)
      out = F.relu(self.AvgMaxPool(out))
      out = out.view(out.size(0),-1)
      out = self.fc1(out)
      # out = self.fc2(out)
      # out = self.fc3(out)
      return out


class IneptionRGSNet16(nn.Module):
    def __init__(self):
        super(IneptionRGSNet16,self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv1d(1, 16, kernel_size=21, padding=0),
                        nn.BatchNorm1d(16),
                        nn.ReLU())
        self.Inception1 = Inception16(in_c=16, c1=20, c2=(10, 20), c3=(10, 20), c4=20, out_C=80)
        # self.Inception2 = Inception(in_c=80, c1=25, c2=(16, 25), c3=(16, 25), c4=25, out_C=130)
        # self.Inception3 = Inception(in_c=130, c1=32, c2=(25, 32), c3=(25, 32), c4=32, out_C=128)
        self.mp = nn.MaxPool1d(3)
        self.AvgMaxPool = nn.AdaptiveMaxPool1d(70)
        self.fc1 = nn.Linear(5600, 1000) #8960 ,7000, 5600
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 1)
        self.drop = nn.Dropout(0.25)

    def forward(self,out):
      out = F.relu(self.conv1(out))
      out = self.Inception1(out)
      # out = self.Inception2(out)
      # out = self.Inception3(out)
      # out = self.conv3(out)
      out = F.relu(self.AvgMaxPool(out))
      out = out.view(out.size(0),-1)
      out = self.fc1(out)
      out = self.drop(out)
      out = self.fc2(out)
      out = self.drop(out)
      out = self.fc3(out)
      return out



class IneptionARGSNet(nn.Module):
    def __init__(self):
        super(IneptionARGSNet,self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=17, padding=0)
        self.Inception1 = InceptionA(in_c=32, c1=20, c2=20, c3=20, c4=20, out_C=80)
        self.Inception2 = InceptionA(in_c=80, c1=25, c2=25, c3=25, c4=25, out_C=100)
        # self.Inception3 = InceptionA(in_c=130, c1=32, c2=32, c3=32, c4=32, out_C=128)
        self.mp = nn.MaxPool1d(3)
        self.AvgMaxPool = nn.AdaptiveMaxPool1d(70)
        self.fc1 = nn.Linear(7000, 1) #8960 ,17920
        # self.fc2 = nn.Linear(1000, 130)
        # self.fc3 = nn.Linear(130, 1)

    def forward(self,out):
      out = F.relu(self.conv1(out))
      # out = self.mp(F.relu(self.conv1(out)))
      # out = self.mp(F.relu(self.conv1(out)))
      out = self.Inception1(out)
      out = self.Inception2(out)
      # out = self.Inception3(out)
      # out = self.conv3(out)
      out = F.relu(self.AvgMaxPool(out))
      out = out.view(out.size(0),-1)
      out = self.fc1(out)
      # out = self.fc2(out)
      # out = self.fc3(out)
      return out


class incep1(nn.Module):
    def __init__(self):
        super(incep1,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=13, stride=1, padding=0),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.incep = Inception(in_c=16, c1=16, c2=(8, 16), c3=(8, 16), c4=16, out_C=64)
        self.AvgMaxPool = nn.AdaptiveMaxPool1d(80)
        self.fc = nn.Linear(3200, 1)

    def forward(self,out):
      out = self.conv1(out)
      out = self.incep(out)
      # out = F.relu(self.AvgMaxPool(out))
      out = self.AvgMaxPool(out)
      out = out.view(out.size(0),-1)
      out = self.fc(out)
      return out

class MSCblock(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, pad=1, relu=True, act=True):
        super(MSCblock, self).__init__()
        self.outplanes = outplanes
        inter_planes = inplanes // 4

        self.conv = nn.Conv1d(outplanes, inter_planes, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(
            nn.Conv1d(inter_planes, inter_planes, 3, stride=stride, padding=pad, bias=False),
            nn.BatchNorm1d(inter_planes),
            #            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv1d(inter_planes, inter_planes, 3, stride=stride, padding=3, bias=False, dilation=3),
            nn.BatchNorm1d(inter_planes)
            #            Mish(inter_planes)
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(inter_planes, inter_planes, 3, stride=stride, padding=5, bias=False, dilation=5),
            nn.BatchNorm1d(inter_planes)
            #             Mish(inter_planes)
        )
        self.branch4 = nn.Sequential(
            nn.Conv1d(inter_planes, inter_planes, 3, stride=stride, padding=7, bias=False, dilation=7),
            nn.BatchNorm1d(inter_planes)
            #             Mish(inter_planes)
        )
        self.relu = nn.ReLU(inplace=True)

    #		self.relu = Mish()
    def forward(self, input):

        input = self.conv(input)
        #		output = self.conv(input)
        x1 = self.branch1(input)

        x1 = self.relu(x1)
        x2 = self.branch2(input)

        x2 = self.relu(x2)
        x3 = self.branch3(input)

        x3 = self.relu(x3)
        x4 = self.branch4(input)

        x4 = self.relu(x4)
        output = torch.cat(((x1 + x2 + x3 + x4), (x1 + x2 + x3 + x4), (x1 + x2 + x3 + x4), (x1 + x2 + x3 + x4)), 1)
        output = self.relu(output)
        return output

class MSCnet(nn.Module):
    def __init__(self):
        super(MSCnet,self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=17, padding=0)
        self.Inception1 = MSCblock(inplanes=32,outplanes=32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=15, padding=0)
        self.Inception2 = MSCblock(inplanes=64,outplanes=64)
        # self.Inception3 = InceptionA(in_c=130, c1=32, c2=32, c3=32, c4=32, out_C=128)
        # self.mp = nn.MaxPool1d(3)
        self.AvgMaxPool = nn.AdaptiveMaxPool1d(150)
        self.fc1 = nn.Linear(9600, 1) #8960 ,17920
        # self.fc2 = nn.Linear(1000, 130)
        # self.fc3 = nn.Linear(130, 1)

    def forward(self,out):
      out = F.relu(self.conv1(out))
      out = self.Inception1(out)
      out = F.relu(self.conv2(out))
      out = self.Inception2(out)
      # out = self.Inception3(out)
      # out = self.conv3(out)
      out = F.relu(self.AvgMaxPool(out))
      out = out.view(out.size(0),-1)
      out = self.fc1(out)
      # out = self.fc2(out)
      # out = self.fc3(out)
      return out


#####16
class Inceptiona16(nn.Module):
    def __init__(self,in_c,c1,c2,c3,c4,out_C):
        super(Inceptiona16,self).__init__()
        self.p1 = nn.Sequential(
            nn.Conv1d(in_c,c1,kernel_size=7,padding=3),
            nn.BatchNorm1d(c1),
            nn.ReLU()
        )
        self.p2 = nn.Sequential(
            nn.Conv1d(in_c,c2,kernel_size=13,padding=6),
            nn.BatchNorm1d(c2),
            nn.ReLU(),
        )
        self.p3 = nn.Sequential(
            nn.Conv1d(in_c, c3, kernel_size=21,padding=10),
            nn.BatchNorm1d(c3),
            nn.ReLU(),
        )
        self.p4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3,stride=1,padding=1),
            nn.Conv1d(in_c,c4,kernel_size=3,padding=1),
            nn.ReLU()
        )
        self.conv_linear = nn.Conv1d((c1+c2+c3+c4), out_C, 1, 1, 0, bias=True)
        self.short_cut = nn.Sequential()
        if in_c != out_C:
            self.short_cut = nn.Sequential(
                nn.Conv1d(in_c, out_C, 1, 1, 0, bias=False),
                # nn.BatchNorm1d(out_C),
                # nn.ReLU()
            )
    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)
        p4 = self.p4(x)
        out =  torch.cat((p1,p2,p3,p4),dim=1)
        # out = self.conv_linear(out)
        out += self.short_cut(x)
        return F.relu(out)

######IncepIDRC20164.pt
class IneptionARGSNet16(nn.Module):
    def __init__(self):
        super(IneptionARGSNet16,self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=21, padding=0)
        self.Inception1 = Inceptiona16(in_c=32, c1=20, c2=20, c3=20, c4=20, out_C=80)
        self.Inception2 = Inceptiona16(in_c=80, c1=25, c2=25, c3=25, c4=25, out_C=100)
        # self.Inception3 = Inceptiona16(in_c=130, c1=25, c2=25, c3=25, c4=25, out_C=130)
        self.mp = nn.MaxPool1d(3)
        # self.AvgMaxPool = nn.AdaptiveMaxPool1d(40)
        self.Avgpool = nn.AdaptiveAvgPool1d(50)
        # self.AvgMaxPool2 = nn.AdaptiveMaxPool1d(50)
        self.drop = nn.Dropout(0.2)
        self.fc1 = nn.Linear(5000, 1) #8960 ,17920
        # self.fc2 = nn.Linear(1000, 130)
        # self.fc3 = nn.Linear(130, 1)

    def forward(self,out):
      out = F.relu(self.conv1(out))
      # out = self.mp(F.relu(self.conv1(out)))
      # out = self.mp(F.relu(self.conv1(out)))
      out = self.Inception1(out)
      out = self.Inception2(out)
      # out = self.Inception3(out)
      # out = self.conv3(out)
      # out = F.relu(self.AvgMaxPool(out))
      out = F.relu(self.Avgpool(out))
      out = out.view(out.size(0),-1)
      out = self.fc1(out)
      # out = self.drop(out)
      # out = self.fc2(out)
      # out = self.drop(out)
      # out = self.fc3(out)
      # out = self.fc2(out)
      # out = self.fc3(out)
      return out

