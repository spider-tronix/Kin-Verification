# python main.py -wd E:\Prem\spiderRD\kaggleFIW -wts E:\Prem\spiderRD\senet50_scratch_weight.pkl -r no -testfn test_labels.csv -trainfn train_labels.csv --arch_type senet50_scratch#
# python main.py -wd E:\Prem\spiderRD\kaggleFIW -r no -testfn test_labels.csv -trainfn train_annotations.csv --arch_type resnet50 -dt imagenet#
# python main.py -wd E:\Prem\spiderRD\kaggleFIW -wts E:\Prem\spiderRD\resnet50_scratch_weight.pkl -r no -testfn test_labels.csv -trainfn train_labels.csv --arch_type resnet50_scratch --check_training subset #
# python main.py -wd E:\Prem\spiderRD\kaggleFIW -r no -testfn test_labels.csv -trainfn train_annotations.csv --arch_type resnet50 -dt imagenet --check_training tuning --loss_fn classification #
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from dataset import FIW, KinFaceW1
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset
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
from check_training import performance
from torch.backends import cudnn
import random
# for hyperparameter tuning
import optuna
from optuna.trial import TrialState
from losses import ContrastiveLoss

# global variables
num_classes = 2
start_fid = "F0600"
end_fid = "F0800"
img_size = 224
N_IDENTITY = 8631

# hyperparameter setting
configurations = {
    1: dict(
        max_iteration=500000,
        lr=1.0e-3,
        momentum=0.74,
        beta1=0.9,
        beta2=0.999,
        weight_decay=0.01,
        gamma=0.1, # "lr_policy: step"
        step_size=75000, # "lr_policy: step"
        interval_validate=2000,
    ),
}

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', help="specify the dataset to use by <-d fiw> or <-d kinface>", default="fiw", choices=['fiw', 'kinface'])
parser.add_argument('-wd', '--working_dir', help="specify the path to your dataset directory by <-wd your_directory>", default="/content/drive/MyDrive/Kinship Verification/fiw")
parser.add_argument('-dt', '--dataset_type', default='vggface2', help='Specify the dataset from which the pretrained model should be loaded from or specify "scratch"',
                    choices=['vggface2', 'imagenet', 'scratch'])
parser.add_argument('-wts', '--weights', help="specify the path to the model weights if you are choosing vggface2 pretrained model")
parser.add_argument('-r', '--resume', default='yes', help='specify whether to resume from a checkpoint file - <yes> or <no>', choices=['yes', 'no'])
parser.add_argument('-cd', '--checkpoint_dir', type=str, default='checkpoints',help='relative path to the checkpoints directory')
parser.add_argument('-c', '--config', type=int, default=1, choices=configurations.keys(),
                    help='the number of settings and hyperparameters used in training')
parser.add_argument('-bs', '--batch_size', type=int, default=32, help='batch size')
parser.add_argument('-testfn', '--testfilename', default="test_labels.csv", help="specify the label filename for test")
parser.add_argument('-trainfn', '--trainfilename', default="train_labels.csv", help="specify the label filename for train")
parser.add_argument('-w', '--workers', default=2, type=int, metavar='N',help='number of data loading workers (default: 4)')
parser.add_argument('--arch_type', type=str, default='resnet50_scratch', help='model type',choices=['resnet50_scratch', 'senet50_scratch', 'resnet50', 'vgg16', 'densenet121'])
parser.add_argument('--check_training', help='run model with a single data point',choices=['single', 'subset', 'tuning'])
parser.add_argument('--loss_fn', help='classification loss or contrastive loss or metric loss', choices=["classification", "contrastive", "metric"], default="classification")

def set_seed(seed):
    """
    Seeds pretty much everything that can be.
    :param seed: the seed number to be used
    :return: None
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

# initialize for scratch training only
def weights_initialize(m):
    if args.dataset_type == "scratch":
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

# optuna objective function
def objective(trial):
    if args.dataset_type == "vggface2":
        include_top = False
        if 'resnet' in args.arch_type:
            model = ResNet.resnet50(num_classes=N_IDENTITY, include_top=include_top) 
        else:
            model = SENet.senet50(num_classes=N_IDENTITY, include_top=include_top)
        utils.load_state_dict(model, args.weights)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = fcNet(in_features=[2048, 512, 128], num_classes=2, trial=trial)           # random weights are automatically initialized

    #other models
    else:
        pretrained = True if args.dataset_type == "imagenet" else False
        model = MyNet(arch=args.arch_type, pretrained=pretrained, trial=trial)
        model.apply(weights_initialize)
    
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    
    params_to_update = model.parameters()
    if args.dataset_type != "scratch":
        params_to_update = []
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
    
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-0, log=True)
    if optimizer_name == "SGD":
        momentum = trial.suggest_uniform("momentum", 0.5, 1.0)
        optim = torch.optim.SGD(
            params_to_update,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum
        )

    else:
        beta1 = trial.suggest_uniform("beta1", 0.5, 1.0)
        beta2 = trial.suggest_uniform("beta2", 0.5, 1.0)
        optim = torch.optim.Adam(
            params_to_update,
            lr = lr,
            weight_decay=weight_decay,
            betas = (beta1,beta2)
        )

    # gamma = trial.suggest_float("gamma", 1e-4, 1.0, log=True)
    # step_size = trial.suggest_int("step_size", )
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, cfg['step_size'],gamma=gamma, last_epoch=last_epoch)

    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    p = performance(
        model=model,
        optim=optim,
        cuda=torch.cuda.is_available(),
        criterion=criterion,
        loader=train_loader,
        val_loader=val_loader,
        batch_size=args.batch_size,
        num_epochs=10,
        operation=args.check_training,
        loss_fn = args.loss_fn,
        trial=trial
    )

    avg_accuracy = p.check_subset()
    return avg_accuracy
    

    
# def get_parameters(model, bias=False):
#     for k, m in model._modules.items():
#         if k == "fc" and isinstance(m, nn.Linear):
#             if bias:
#                 yield m.bias
#             else:
#                 yield m.weight

if __name__ == "__main__":
    args = parser.parse_args()
    if args.dataset_type == "vggface2" and not args.weights:
        print("Specify the path to the weights")
        sys.exit()
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
    log_file = f"logger_{args.dataset_type}_{args.arch_type}.txt"
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

    if args.check_training is not None:
        random.seed(10)
        l1 = random.sample(range(len(train_data)), int(0.01*len(train_data)))
        train_subset = Subset(train_data, l1)
        if args.dataset == "fiw":
            random.seed(20)
            l2 = random.sample(range(1,len(val_data)), int(0.01*len(val_data)))
            val_subset = Subset(val_data, l2)
            
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, **kwargs)
        if val_data:
            #val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, **kwargs)
            val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, **kwargs)

    else:
        random.seed(10)
        l1 = random.sample(range(len(train_data)), int(0.001*len(train_data)))
        train_subset = Subset(train_data, l1)
        if args.dataset == "fiw":
            random.seed(20)
            l2 = random.sample(range(1,len(val_data)), int(0.001*len(val_data)))
            val_subset = Subset(val_data, l2)
            
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, **kwargs)
        if val_data:
            val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, **kwargs)
            #val_loader = DataLoader(val_subset, batch_size=len(l2), shuffle=False, **kwargs)
        # train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **kwargs)

        # if val_data:
        #     val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, **kwargs)

    #test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, **kwargs)

    # print(len(train_loader), len(val_loader))
    # finetuning
    if args.check_training == "tuning":
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=10)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
        sys.exit()
    
    
    
    #model vggface2
    if args.dataset_type == "vggface2":
        include_top = False
        if 'resnet' in args.arch_type:
            model = ResNet.resnet50(num_classes=N_IDENTITY, include_top=include_top) 
        else:
            model = SENet.senet50(num_classes=N_IDENTITY, include_top=include_top)

        utils.load_state_dict(model, args.weights)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = fcNet(in_features=[2048, 512, 128], num_classes=2)           # random weights are automatically initialized
        
    #other models
    else:
        pretrained = True if args.dataset_type == "imagenet" else False
        model = MyNet(arch=args.arch_type, pretrained=pretrained, loss_fn=args.loss_fn)
        model.apply(weights_initialize)

    # training setting
    checkpoint_file = f"checkpoints_{args.dataset_type}_{args.arch_type}.pth.tar"
    if not os.path.exists(args.checkpoint_dir):
        utils.create_dir(args.checkpoint_dir)
    start_epoch = 0
    start_iteration = 0
    if resume == "yes" or resume == "y" or resume == "Yes":
        f = os.path.join(args.checkpoint_dir, checkpoint_file)
        checkpoint = torch.load(f)

        if args.dataset_type == "scratch":
            model.load_state_dict(checkpoint['model_state_dict'])
        elif args.dataset_type == "vggface2":
            model.fc.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.fnet.load_state_dict(checkpoint['model_state_dict'])

        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
        # if args.dataset_type == "vggface2":
        #     assert checkpoint['arch'] == args.arch_type
        print("Resume from epoch: {}, iteration: {}".format(start_epoch, start_iteration))

    if torch.cuda.is_available():
        model = model.cuda()

    # criterion
    if args.loss_fn != "contrastive":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = ContrastiveLoss()
    
    if torch.cuda.is_available():
        criterion = criterion.cuda()

    params_to_update = model.parameters()
    if args.dataset_type != "scratch":
        params_to_update = []
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)

    # SGD optimizer
    optim = torch.optim.SGD(
        params_to_update,
        lr=cfg['lr'],
        momentum=cfg['momentum'],
        weight_decay=cfg['weight_decay'])

    # optim = torch.optim.Adam(
    #     params_to_update,
    #     lr=cfg['lr'],
    #     betas=(cfg['beta1'], cfg['beta2'])
    # )
    if resume == "yes" or resume == "Yes" or resume == "y":
        optim.load_state_dict(checkpoint['optim_state_dict'])
        
    # lr_policy: step
    last_epoch = start_iteration if resume == "yes" or resume == "Yes" or resume == "y"  else -1
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, cfg['step_size'],gamma=cfg['gamma'], last_epoch=last_epoch)

    # for name, param in model.named_parameters():
    #     print(name, ':', param.requires_grad)
    # torch.save(model.fnet.state_dict(), 'weights_only.pth')
    # model_new.fnet.load_state_dict(torch.load('weights_only.pth'))
    # print("\nloaded\n")
    # for name, param in model_new.named_parameters():
    #     print(name, ':', param.requires_grad)
    # sys.exit()

    # train the model
    print(model.fnet)
    tb_dir = os.path.join("runs", args.dataset_type, args.arch_type)
    if args.check_training is not None and args.check_training != "tuning":
        p = performance(
            model=model,
            optim=optim,
            cuda=torch.cuda.is_available(),
            criterion=criterion,
            loader=train_loader,
            val_loader=val_loader,
            batch_size=args.batch_size,
            num_epochs=100,
            operation=args.check_training,
            loss_fn=args.loss_fn,
            lr_scheduler=lr_scheduler
        )
        # check_subset(model, optim, torch.cuda.is_available(), criterion, train_loader, args.batch_size, lr_scheduler)
        sys.exit()
    trainer = Trainer(
        arch_type = args.arch_type,
        dataset = args.dataset,
        dataset_type = args.dataset_type,
        cmd="train",
        cuda=torch.cuda.is_available(),
        model=model,
        criterion=criterion,
        optimizer=optim,
        lr_scheduler=lr_scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        test_data=test_data,
        log_file=log_file,
        max_iter=cfg['max_iteration'],
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_file = checkpoint_file,
        print_freq=500,
        interval_validate = 4,#cfg["interval_validate"],
        tb_dir = tb_dir,
        loss_fn = args.loss_fn
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    print("training...")
    trainer.train()