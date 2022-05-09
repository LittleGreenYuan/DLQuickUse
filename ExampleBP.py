# -*- coding: utf-8 -*-
"""
@author: Greenyuan
"""
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from scipy.io import loadmat as load#调用load进行.mat文件的读取

#继承于torch的自定义数据集
class DateSet(Dataset):
    def __init__(self,data,label,transform=None):
        self.data = data
        self.label = label
        self.transform = transform
    def __getitem__(self,index):
        return self.data[index],self.label[index]
    def __len__(self):
        return len(self.data)
#继承于nn.Module的自定义网络结构，此处定义的是BP神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.net1 = nn.Sequential(nn.Linear(24,32),nn.ReLU())
        self.net2= nn.Sequential(nn.Linear(32,2))#输出最后要与标签对应，示例数据集是1位结果
    def forward(self,x):
        x=self.net1(x)
        x=x.view(x.size(0),-1)
        out = self.net2(x)
        return out
        
def readData():
    #自定义自己的数据加载，返回的是数据集与标签
    DATA_c1 = load('dataset\\BPdata\\data1.mat')#读取数据，其中关键标签c1对应数据集，第一列为标签，第二列为数据
    data_c1=np.zeros(DATA_c1['c1'].shape)#初始化定义矩阵
    data_c1[:,:] = DATA_c1['c1']
    
    DATA_c2 = load('dataset\\BPdata\\data2.mat')#读取数据，其中关键标签c2对应数据集，第一列为标签，第二列为数据
    data_c2=np.zeros(DATA_c2['c2'].shape)#初始化定义矩阵
    data_c2[:,:] = DATA_c2['c2']
    
    train=np.vstack([data_c1[:,1:DATA_c1['c1'].shape[1]], data_c2[:,1:DATA_c2['c2'].shape[1]]])#将数据上下连接
    label_temp=np.vstack([data_c1[:,0].reshape([DATA_c1['c1'].shape[0],1]),data_c2[:,0].reshape([DATA_c2['c2'].shape[0],1])])#将标签上下连接
    label=np.array([0,0])
    #将标签转换成热占位码
    for i in range(len(label_temp)):
        if(label_temp[i]==1):
            label=np.vstack([label,np.array([[0.,1.]])])
        elif(label_temp[i]==2):
            label=np.vstack([label,np.array([[1.,0.]])])
    label=np.delete(label,0,axis=0)
    return train ,label


def train():
    
    use_cuda=torch.cuda.is_available()
    data,label=readData()
    train_data=DateSet(data,label,transform=transforms.ToTensor())#定义训练数据
    val_data=DateSet(data,label,transform=transforms.ToTensor())#定义测试数据（在这里方便也用了训练数据）
    train_loader = DataLoader(dataset=train_data,batch_size=32,shuffle=True)#分批次送入训练数据
    val_loader = DataLoader(dataset=train_data,batch_size=32)#分批次送入测试s数据
    
    model = Net()#定义网络实体
    model = model.double()
    
    optimizer = torch.optim.Adam(model.parameters(),lr=0.01,weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,[10,20],0.1)
    loss_func = nn.CrossEntropyLoss()
    
    for epoch in range(20):#总训练次数
        model.train()
        loss_train=0
        for batch, (batch_x,batch_y) in enumerate(train_loader):#分批次输入训练样本
            batch_x, batch_y = Variable(batch_x), Variable(batch_y)
            out = model(batch_x)
            loss = loss_func(out,batch_y)
            #print(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train = loss_train+loss.item()
        scheduler.step()
        print("Loss: "+str(loss_train))
        
        model.eval()
        for batch_x, batch_y in val_loader:#测试训练模型
            batch_x, batch_y = Variable(batch_x), Variable(batch_y)
            out = model(batch_x)
        if((epoch+1)%20==0):#保存模型
            torch.save(model.state_dict(), 'params_'+ str(epoch+1)+'.pth')
            Outpth='params_'+ str(epoch+1)+'.pth'
            print('保存了模型')
    return Outpth

def transformONNX(Outpth):
    #对torch保存的模型进行onnx转换
    #网络的再次重新定义
    model_test = Net()#定义网络实体
    model_test =model_test.double()
    model_statedict = torch.load(Outpth,map_location=lambda storage,loc:storage)   #导入Gpu训练模型，导入为cpu格式
    model_test.load_state_dict(model_statedict)  #将参数放入model_test中
    model_test.eval()  # 测试，看是否报错
    #下面开始转模型，cpu格式下
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_input = torch.randn(1,24,device=device).double()
    input_names = ["input"]
    output_names = ["output"]
    torch.onnx.export(model_test, dummy_input, "model_.onnx", opset_version=None, verbose=False, output_names=["output"])

if __name__ == '__main__':
    Outpth=train()
    transformONNX(Outpth)
    

