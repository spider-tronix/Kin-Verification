# python main.py -wd E:\Prem\spiderRD\kaggleFIW -wts E:\Prem\spiderRD\senet50_scratch_weight.pkl -r no -testfn test_labels.csv -trainfn train_labels.csv --arch_type senet50_scratch#
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from dataset import FIW, KinFaceW1
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import models.resnet as ResNet
import models.senet as SENet
from models.baseline import MyNet
from models.baseline import fcNet
import utils
import math
import argparse
import sys
from trainer import Trainer

# global variables
num_classes = 2
start_fid = "F0500"
end_fid = "F0550"
img_size = 224
N_IDENTITY = 8631

# hyperparameter setting
configurations = {
    1: dict(
        max_iteration=62500,
        lr=1.0e-1,
        momentum=0.9,
        weight_decay=0.0,
        gamma=0.1, # "lr_policy: step"
        step_size=20, # "lr_policy: step"
        interval_validate=1000,
    ),
}

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', help="specify the dataset to use by <-d fiw> or <-d kinface>", default="fiw", choices=['fiw', 'kinface'])
parser.add_argument('-wd', '--working_dir', help="specify the path to your dataset directory by <-wd your_directory>", default="/content/drive/MyDrive/Kinship Verification/fiw")
parser.add_argument('-dt', '--dataset_type', default='vggface2', help='Specify the dataset from which the pretrained model should be loaded from or specify "scratch"',
                    choices=['vggface2', 'imagenet', 'scratch'])
parser.add_argument('-wts', '--weights', help="specify the relative path to the model weights if you are choosing vggface2 pretrained model")
parser.add_argument('-r', '--resume', default='yes', help='specify whether to resume from a checkpoint file - <yes> or <no>', choices=['yes', 'no'])
parser.add_argument('-cd', '--checkpoint_dir', type=str, default='checkpoints',help='relative path to the checkpoints directory')
parser.add_argument('-cf', '--checkpoint_file', type=str, default='',help='checkpoint file name')
parser.add_argument('-c', '--config', type=int, default=1, choices=configurations.keys(),
                    help='the number of settings and hyperparameters used in training')
parser.add_argument('-bs', '--batch_size', type=int, default=32, help='batch size')
parser.add_argument('-testfn', '--testfilename', default="test_labels.csv", help="specify the label filename for test")
parser.add_argument('-trainfn', '--trainfilename', default="train_labels.csv", help="specify the label filename for train")
parser.add_argument('-w', '--workers', default=4, type=int, metavar='N',help='number of data loading workers (default: 4)')
parser.add_argument('--arch_type', type=str, default='resnet50_scratch', help='model type',choices=['resnet50_scratch', 'senet50_scratch', 'resnet50', 'vgg16', 'densenet161'])
parser.add_argument('-logfn', '--log_file', type=str, default='logger.log', help='log file')

args = parser.parse_args()
if args.dataset_type == "vggface2" and not args.weights:
    print("Specify the path to the weights")
    sys.exit

# initialize for random weights only
def weights_initialize(m):
    if args.dataset_type == "scratch":
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

def get_parameters(model, bias=False):
    for k, m in model._modules.items():
        if k == "fc" and isinstance(m, nn.Linear):
            if bias:
                yield m.bias
            else:
                yield m.weight


device = "cuda" if torch.cuda.is_available() else "cpu"
resume = args.resume
# transforms
my_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.CenterCrop(img_size),
        transforms.Normalize((127.5,127.5,127.5), (127.5,127.5,127.5))
    ]
)

kwargs = {'num_workers': args.workers, 'pin_memory': True} if torch.cuda.is_available() else {}

# dataloader
root = args.working_dir
log_file = args.log_file
cfg = configurations[args.config]
label_train = args.trainfilename
label_test = args.testfilename

if args.dataset == "fiw":
    fids = next(os.walk(os.path.join(root,"train")))[1]
    val_list = [fid for fid in fids if (fid>=start_fid and fid<=end_fid)]
    train_data = FIW(root, label_train, "train", my_transforms, val_list)
    val_data = FIW(root, label_train, "val", my_transforms, val_list)
    test_data = FIW(root, label_test, "test", my_transforms)

else:
    train_data = KinFaceW1(root, "train", my_transforms)
    test_data = FIW(root, "test", my_transforms)

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **kwargs)
if val_data:
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, **kwargs) 
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, **kwargs)

#model vggface2
if args.dataset_type == "vggface2":
    include_top = False
    if 'resnet' in args.arch_type:
        model = ResNet.resnet50(num_classes=N_IDENTITY, include_top=include_top)
    else:
        model = SENet.senet50(num_classes=N_IDENTITY, include_top=include_top)

    if resume != "yes" or resume != "y" or resume != "Yes":
        utils.load_state_dict(model, args.weights)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = fcNet(in_features=[2048, 512, 128], num_classes=2)           # random weights are automatically initialized
        #model.fc.reset_parameters()
#other models
else:
    pretrained = True if args.dataset_type == "imagenet" else False
    model = MyNet(arch=args.arch_type, pretrained=pretrained)
    model.apply(weights_initialize)

# training setting
start_epoch = 0
start_iteration = 0
if resume == "yes" or resume == "y" or resume == "Yes":
    f = os.path.join(root, args.checkpoint_dir, args.checkpoint_file)
    checkpoint = torch.load(f)
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    start_iteration = checkpoint['iteration']
    if args.dataset_type == "vggface2":
        assert checkpoint['arch'] == args.arch_type
    print("Resume from epoch: {}, iteration: {}".format(start_epoch, start_iteration))

if torch.cuda.is_available():
    model = model.cuda()

# criterion
criterion = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    criterion = criterion.cuda()

# SGD optimizer
optim = torch.optim.SGD(
    [
        {'params': get_parameters(model, bias=False)},
        {'params': get_parameters(model, bias=True), 'lr': cfg['lr'] * 2, 'weight_decay': 0},
    ],
    lr=cfg['lr'],
    momentum=cfg['momentum'],
    weight_decay=cfg['weight_decay'])
if resume == "yes" or resume == "Yes" or resume == "y":
    optim.load_state_dict(checkpoint['optim_state_dict'])
    
# lr_policy: step
last_epoch = start_iteration if resume == "yes" or resume == "Yes" or resume == "y"  else -1
lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, cfg['step_size'],gamma=cfg['gamma'], last_epoch=last_epoch)

# train the model
trainer = Trainer(
    dataset = args.dataset,
    cmd="train",
    cuda=torch.cuda.is_available(),
    model=model,
    criterion=criterion,
    optimizer=optim,
    lr_scheduler=lr_scheduler,
    train_loader=train_loader,
    val_loader=val_loader,
    log_file=log_file,
    max_iter=cfg['max_iteration'],
    checkpoint_dir=args.checkpoint_dir,
    print_freq=1,
)
trainer.epoch = start_epoch
trainer.iteration = start_iteration
print("training...")
trainer.train()
