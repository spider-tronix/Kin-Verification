import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, arch, pretrained=False):
        super(ConvNet, self).__init__()
        self.arch = arch
        self.num_classes = num_classes
        self.pretrained = pretrained
        
        if self.arch=="resnet50":
            original_model = torchvision.models.resnet50(pretrained=self.pretrained)
        elif self.arch=="vgg16":
            original_model = torchvision.models.vgg16(pretrained=self.pretrained)
        if self.arch=="densenet161":
            original_model = torchvision.models.densenet161(pretrained=self.pretrained)
        
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        if self.pretrained:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x, y):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.avg_pool2d(x, kernel_size=7).view(x.size(0), -1)

        y = self.features(y)
        y = F.relu(y, inplace=True)
        y = F.avg_pool2d(y, kernel_size=7).view(y.size(0), -1)
        return x, y

class fcNet(nn.Module):
    def __init__(self, in_features, num_classes=2):
        super(fcNet, self).__init__()
        self.in_features = in_features
        self.dens1 = nn.Linear(in_features=self.in_features[0]*2, out_features=self.in_features[0])
        self.dens2 = nn.Linear(in_features=self.in_features[0], out_features=self.in_features[1])
        self.dens3 = nn.Linear(in_features=self.in_features[1], out_features=self.in_features[2])
        
        if len(self.in_features)>3:
            self.dense5 = nn.Linear(in_features=self.in_features[2], out_features=self.in_features[3])
        self.dens4 = torch.nn.Linear(in_features=self.in_features[-1], out_features=num_classes)

    def forward(self, x):
        x = self.dens1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.25, training=self.training)
        
        
        x = self.dens2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.25, training=self.training)
        
        x = self.dens3(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.25, training=self.training)

        if len(self.in_features)>3:
            x = self.dense5(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.25, training=self.training)

        x = self.dens4(x)
        return x

class MyNet(torch.nn.Module):
    def __init__(self, arch, pretrained=False):
        super().__init__()
        self.arch = arch
        self.pretrained = pretrained

        if self.arch == "densenet161":
            self.in_features = [2208, 512, 128]
        elif self.arch == "resnet50":
            self.feature_dims = [2048, 512, 128]
        else:
            self.in_features = [25088, 4096, 4096, 512, 128]
        
        self.cnet = ConvNet(self.arch, self.pretrained)
        self.fnet = fcNet(self.in_features)
    
    def forward(self, x, y):
        x, y = self.cnet(x, y)
        v = torch.cat((x,y), dim=1)
        v = self.fnet(v)
        return v

