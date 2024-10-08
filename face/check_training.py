import torch
import torch.nn as nn
from torch.utils.data import dataloader
import numpy as np
from torch.autograd import Variable
import random
import optuna
import torch.nn.functional as F

threshold = 0.3

class performance(object):
    def __init__(self, model, optim, cuda, criterion, loader, val_loader, batch_size, num_epochs=10, lr_scheduler=None, operation="subset", trial=None, loss_fn="contrastive"):
        self.model = model
        self.optim = optim
        self.cuda = cuda
        self.criterion=criterion
        self.loader = loader
        self.batch_size = batch_size
        self.lr_scheduler = lr_scheduler
        self.trial = trial
        self.num_epochs = num_epochs
        self.operation = operation
        self.val_loader = val_loader
        self.loss_fn = loss_fn

        if self.operation == "single":
            self.check_single()
        elif self.operation == "subset":
            self.check_subset()

    def check_single(self):
        self.model.train()
        self.optim.zero_grad()
        imgs1, imgs2, ages1, ages2, genders1, genders2, target = next(iter(self.loader))
        target = target.long()
        #target = target.view(-1).long()
        index = random.randint(0,batch_size)
        imgs1 = imgs1[index]
        imgs1 = torch.unsqueeze(imgs1,0)
        imgs2 = imgs2[index]
        imgs2 = torch.unsqueeze(imgs2,0)
        target = target[index]
        print("checking training for single data point...")

        if self.cuda:
            imgs1, imgs2, target = imgs1.cuda(), imgs2.cuda(), target.cuda(non_blocking=True)
        imgs1, imgs2, target = Variable(imgs1), Variable(imgs2), Variable(target)

        for epoch in range(self.num_epochs):
            output = self.model(imgs1,imgs2)
            loss = self.criterion(output, target)

            if np.isnan(float(loss.data)):
                raise ValueError('loss is nan while training')

            _, prediction = output.max(1)
            num_correct = (prediction == target).sum()
        
            print(
                    f"Epoch: {epoch} Loss : {loss:.4f} Prediction: {prediction} Target: {target}\n"
                )

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

    def check_subset(self):
        print("training a subset...")
        var_name = 'param_groups'
        self.model.train()
        self.optim.zero_grad()
        for epoch in range(self.num_epochs):
            loss_sum = torch.Tensor([0])
            acc_sum = torch.Tensor([0])
            n_batches = 0
            for batch_idx, (imgs1, imgs2, ages1, ages2, genders1, genders2, target) in enumerate(self.loader):
                iteration = batch_idx + epoch * len(self.loader)
                target = target.view(-1).long()

                if self.cuda:
                    imgs1, imgs2, target = imgs1.cuda(), imgs2.cuda(), target.cuda(non_blocking=True)
                imgs1, imgs2, target = Variable(imgs1), Variable(imgs2), Variable(target)

                if self.loss_fn != "contrastive":
                    output = self.model(imgs1, imgs2)
                    loss = self.criterion(output, target)
                else:
                    output1, output2 = self.model(imgs1, imgs2)
                    dist = F.pairwise_distance(output1, output2)
                    dist = dist < threshold
                    loss = self.criterion(output1, output2, target)
                if np.isnan(float(loss.data)):
                    raise ValueError('loss is nan while training')

                if self.loss_fn != "contrastive":
                    _, prediction = output.max(1)
                    num_correct = (prediction == target).sum()
                else:
                    prediction = dist
                    target = target == 1
                    num_correct = (prediction == target).sum()
                acc = float(num_correct)/float(self.batch_size)*100
                n_batches = n_batches + 1

                loss_sum = loss_sum + loss
                acc_sum = acc_sum + acc
                
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()  # update lr

                if (batch_idx) % (len(self.loader)//3) == 0:
                    print(
                        f"Epoch: {epoch}/{self.num_epochs} Batch {batch_idx}/{len(self.loader)} \
                        Current Loss : {float(loss):.4f} Avg loss: {float(loss_sum)/n_batches:.4f} \
                        Cuurent accuracy: {float(acc):.4f} Avg accuracy: {float(acc_sum)/n_batches:.4f}" 
                    )
            
            print(f"finished epoch {epoch}...")
            print(f"Avg Loss: {float(loss_sum)/n_batches:.4f} Avg Accuracy: {float(acc_sum)/n_batches:.4f}")
            self.avg_accuracy = self.validate()
            #print(self.model.training)
            if self.trial is not None:
                # self.avg_accuracy = float(acc_sum)/n_batches
                # self.avg_loss = float(loss_sum)/n_batches
                self.trial.report(self.avg_accuracy, epoch)

                # Handle pruning based on the intermediate value.
                if self.trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
        
        if self.trial is not None:
            return self.avg_accuracy

    def validate(self):
        self.model.eval()
        #print(self.model.training)
        loss_sum = torch.Tensor([0])
        acc_sum = torch.Tensor([0])
        n_batches = 0
        for batch_idx, (imgs1, imgs2, ages1, ages2, genders1, genders2, target) in enumerate(self.val_loader):
            iteration = batch_idx
            target = target.view(-1).long()

            if self.cuda:
                imgs1, imgs2, target = imgs1.cuda(), imgs2.cuda(), target.cuda(non_blocking=True)
            imgs1, imgs2, target = Variable(imgs1), Variable(imgs2), Variable(target)

            
            if self.loss_fn != "contrastive":
                output = self.model(imgs1, imgs2)
                loss = self.criterion(output, target)
            else:
                output1, output2 = self.model(imgs1, imgs2)
                dist = F.pairwise_distance(output1, output2)
                dist = dist < threshold
                loss = self.criterion(output1, output2, target)
            if np.isnan(float(loss.data)):
                raise ValueError('loss is nan while training')

            if self.loss_fn != "contrastive":
                _, prediction = output.max(1)
                num_correct = (prediction == target).sum()
            else:
                prediction = dist
                target = target == 1
                num_correct = (prediction == target).sum()
            acc = float(num_correct)/float(self.batch_size)*100
            n_batches = n_batches + 1
            #print(prediction, target, num_correct)

            loss_sum = loss_sum + loss
            acc_sum = acc_sum + acc
            # if (batch_idx) % (len(self.val_loader)//3) == 0:
            #     print(
            #         f"Batch {batch_idx}/{len(self.val_loader)} \
            #         Current val Loss : {float(loss):.4f} Avg val loss: {float(loss_sum)/n_batches:.4f} \
            #         Cuurent val accuracy: {float(acc):.4f} Avg val accuracy: {float(acc_sum)/n_batches:.4f}" 
            #     )
            
        print(f"finished validating...")
        print(f"Avg val Loss: {float(loss_sum)/n_batches:.4f} Avg val Accuracy: {float(acc_sum)/n_batches:.4f}")

        self.model.train()
        return float(acc_sum)/n_batches