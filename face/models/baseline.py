import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import optuna

class ConvNet(nn.Module):
    def __init__(self, arch, pretrained=False):
        super(ConvNet, self).__init__()
        self.arch = arch
        self.pretrained = pretrained
        
        if self.arch=="resnet50":
            original_model = torchvision.models.resnet50(pretrained=self.pretrained)
        elif self.arch=="vgg16":
            original_model = torchvision.models.vgg16(pretrained=self.pretrained)
            original_model.classifier = nn.Sequential(*list(original_model.classifier.children())[:-1])
        if self.arch=="densenet121":
            original_model = torchvision.models.densenet121(pretrained=self.pretrained)
        
        if self.arch != "vgg16":
            self.features = nn.Sequential(*list(original_model.children())[:-1])
        else:
            self.features = original_model
        if self.pretrained:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x, y):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        if self.arch == "densenet121":
            x = F.avg_pool2d(x, kernel_size=7).view(x.size(0), -1)
        else:
            x = x.view(x.size(0), -1)

        y = self.features(y)
        y = F.relu(y, inplace=True)
        if self.arch == "densenet121":
            y = F.avg_pool2d(y, kernel_size=7).view(y.size(0), -1)
        else:
            y = y.view(y.size(0), -1)
        return x, y

class fcNet(nn.Module):
    def __init__(self, in_features, num_classes=2, trial=None, loss_fn=None):
        super(fcNet, self).__init__()
        self.in_features = in_features
        self.layers = []
        self.trial = trial
        if self.trial is None:
            if loss_fn=="classification":
                # self.model = nn.Linear(self.in_features[0]*2, num_classes)
                self.model = nn.Sequential(nn.Linear(self.in_features[0]*2,self.in_features[-2]),
                                           nn.ReLU(),
                                           nn.Dropout(0.25),

                                           nn.Linear(self.in_features[-2],self.in_features[-1]),
                                           nn.ReLU(),
                                           nn.Dropout(0.25),
                                           nn.Linear(self.in_features[-1], num_classes))
            else:
                self.model = nn.Linear(self.in_features[0], num_classes)
            # self.dens1 = nn.Linear(in_features=self.in_features[0]*2, out_features=self.in_features[0])
            # self.dens2 = nn.Linear(in_features=self.in_features[0], out_features=self.in_features[1])
            # self.dens3 = nn.Linear(in_features=self.in_features[1], out_features=self.in_features[2])
            # self.dens4 = torch.nn.Linear(in_features=self.in_features[-1], out_features=num_classes)
            # self.dens1 = nn.Linear(self.in_features[0]*2, 512)
            # self.dens2 = nn.Linear(512, 32)
            # self.dens3 = nn.Linear(32, num_classes)
        else:
            n_layers = self.trial.suggest_int("n_layers", 0, 2)
            self.in_features = self.in_features[0]*2
            for i in range(n_layers):
                self.out_features = self.trial.suggest_int("n_units_l{}".format(i), 2, 9)
                self.out_features = 2**self.out_features
                self.layers.append(nn.Linear(self.in_features, self.out_features))
                self.layers.append(nn.ReLU())
                p = self.trial.suggest_float("dropout_l{}".format(i), 0.1, 0.5)
                self.layers.append(nn.Dropout(p))

                self.in_features = self.out_features
            self.layers.append(nn.Linear(self.in_features, num_classes))
            self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        if self.trial is None:
            x = self.model(x)
            # x = self.dens1(x)
            # x = F.relu(x)
            # x = F.dropout(x, p=0.25, training=self.training)
            
            
            # x = self.dens2(x)
            # x = F.relu(x)
            # x = F.dropout(x, p=0.25, training=self.training)

            # x = self.dens3(x)
            
            # x = self.dens3(x)
            # x = F.relu(x)
            # x = F.dropout(x, p=0.25, training=self.training)

            # x = self.dens4(x)

        else:
            x = self.model(x)
        return x

class MyNet(torch.nn.Module):
    def __init__(self, arch, trial=None, pretrained=False, loss_fn="classification"):
        super().__init__()
        self.arch = arch
        self.pretrained = pretrained
        self.trial = trial
        self.loss_fn=loss_fn
        if self.arch == "densenet121":
            self.in_features = [1024, 512, 128]
        elif self.arch == "resnet50":
            self.in_features = [2048, 512, 128]
        else:
            self.in_features = [4096, 512, 128]
        
        self.cnet = ConvNet(self.arch, self.pretrained)
        if self.loss_fn != "contrastive":
            self.fnet = fcNet(self.in_features, trial=self.trial, loss_fn=loss_fn)
        else:
            self.lin = nn.Linear(in_features=self.in_features[0], out_features=128)
    
    def forward(self, x, y):
        x, y = self.cnet(x, y)
        
        if self.loss_fn == "contrastive":
            x = self.lin(x)
            y = self.lin(y)
            return x, y

        elif self.loss_fn == "metric":
            v = (x-y)**2
            v = F.normalize(v,dim=-1)
        
        else:
            v = torch.cat((x,y), dim=1)
        
        v = self.fnet(v)
        return v

