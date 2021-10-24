import torch.nn as nn
import torch 
import torchvision


class Gait3d(nn.Module):
    def __init__(self,no_classes):
        super(Gait3d,self).__init__()
        '''
        3D CNN Model proposed as per the paper "KinGaitWild"
        '''
        self.classes=no_classes
        self.conv_layer1=nn.Sequential(nn.Conv3d(in_channels=3,out_channels=32,kernel_size=3,padding=1),nn.ReLU(inplace=True),
                                    nn.Conv3d(in_channels=32,out_channels=32,kernel_size=3,padding=1),nn.ReLU(inplace=True))
        self.pool_layer1=nn.MaxPool3d(kernel_size=2,stride=(4,4,4))
        self.conv_layer2=nn.Sequential(nn.Conv3d(in_channels=32,out_channels=64,kernel_size=3,padding=1),nn.ReLU(inplace=True),
                                    nn.Conv3d(in_channels=64,out_channels=64,kernel_size=3,padding=1),nn.ReLU(inplace=True))
        self.fc1=nn.Sequential(nn.Linear(126464,4096),nn.ReLU(inplace=True),nn.Dropout(0.25))
        self.fc2=nn.Sequential(nn.Linear(4096,4096),nn.ReLU(inplace=True),nn.Dropout(0.25))
        self.fc3=nn.Sequential(nn.Linear(4096,1024),nn.ReLU(inplace=True),nn.Dropout(0.25))
        self.fc4=nn.Sequential(nn.Linear(1024,256),nn.ReLU(inplace=True),nn.Dropout(0.25))
        self.fc5=nn.Sequential(nn.Linear(512,256),nn.ReLU(inplace=True))
        self.fc6=nn.Sequential(nn.Linear(256,no_classes))
    
    def forward_pass(self,img):
        x=self.conv_layer1(img)
        print(x.shape)
        #x=self.conv_layer1(x)
        y=self.pool_layer1(x)
        print(y.shape)
        #x=self.conv_layer2(x)
        embedding=self.conv_layer2(y)
        print(embedding.shape)
        embedding=embedding.reshape(embedding.size(0),-1)
        out=self.fc1(embedding)
        out=self.fc2(out)
        out=self.fc3(out)
        out=self.fc4(out)
        return out
    
    def forward(self,inp1,inp2=None,finetune=False):
        if finetune:
            feature1=self.forward_pass(inp1).squeeze()
            feature2=self.forward_pass(inp2).squeeze()
            out=self.fc5(torch.cat([feature1,feature2]))
            res=self.fc6(out)
        else:
            feature1=self.forward_pass(inp1).squeeze()
            res=self.fc6(feature1)
        return res



if __name__ == "__main__":
    Model=Gait3d(2)
    print(Model)
    input1=torch.rand(size=(1,3,16,150,50))
    print(input1)
    input2=torch.rand(size=(1,3,16,150,50))
    #print(input2)
    prediction=Model(input1,input2,finetune=True)
    print(prediction)
