import datetime
import math
import os
import shutil
import gc
import time

import numpy as np
import torch
from torch.autograd import Variable

import utils
import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import RandomSampler, BatchSampler
from torch.utils.data import DataLoader

class Trainer(object):

    def __init__(self, arch_type, dataset, dataset_type, cmd, cuda, model, criterion, optimizer,
                 train_loader, val_loader, test_data, log_file, max_iter,
                 checkpoint_dir, checkpoint_file, tb_dir,
                 interval_validate=None, lr_scheduler=None,
                 print_freq=1):
        """
        :param cuda:
        :param model:
        :param optimizer:
        :param train_loader:
        :param val_loader:
        :param log_file: log file name. logs are appended to this file.
        :param max_iter:
        :param interval_validate:
        :param checkpoint_dir:
        :param checkpoint_file:
        :param tb_dir:  tensorboard directory
        :param lr_scheduler:
        """


        self.cmd = cmd
        self.cuda = cuda

        self.model = model
        self.criterion = criterion
        self.optim = optimizer
        self.lr_scheduler = lr_scheduler

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_data = test_data
        self.dataset = dataset
        self.dataset_type = dataset_type
        self.arch_type = arch_type 

        self.timestamp_start = datetime.datetime.now()

        if cmd == 'train':
            self.interval_validate = len(self.train_loader) if interval_validate is None else interval_validate

        self.epoch = 0
        self.iteration = 0

        self.max_iter = max_iter
        self.best_top = 0
        self.print_freq = print_freq

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = checkpoint_file
        self.log_file = log_file
        self.tb_dir = tb_dir
        self.step=0
        self.step_val = 0
        self.step_test = 0


    def print_log(self, log_str):
        with open(self.log_file, 'a') as f:
            f.write(log_str + '\n')


    def validate(self):
        batch_time = utils.AverageMeter()
        losses = utils.AverageMeter()
        top = utils.AverageMeter()

        training = self.model.training
        self.model.eval()

        end = time.time()
        if self.dataset == "fiw":                                                                       # different dataloaders
            for batch_idx, (imgs1, imgs2, ages1, ages2, genders1, genders2, target) in tqdm.tqdm(
                enumerate(self.val_loader), total=len(self.val_loader),
                desc='Valid iteration={} epoch={}'.format(self.iteration, self.epoch), ncols=80, leave=True, position=0):
        

                gc.collect()
                target = target.view(-1).long()
                if self.cuda:
                    imgs1, imgs2, target = imgs1.cuda(), imgs2.cuda(), target.cuda(non_blocking=True)          # we have two images
                imgs1 = Variable(imgs1)
                imgs2 = Variable(imgs2)
                target = Variable(target)
                with torch.no_grad():
                    output = self.model(imgs1, imgs2)
                    loss = self.criterion(output, target)

                if np.isnan(float(loss.data)):
                    raise ValueError('loss is nan while validating')

                # measure accuracy and record loss
                prec = utils.accuracy(output.data, target.data)
                losses.update(loss.data, imgs1.size(0))
                top.update(prec[0], imgs1.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                if batch_idx % self.print_freq == 0:
                    # log_str = 'Test: [{0}/{1}/{top.count:}]\tepoch: {epoch:}\titer: {iteration:}\t' \
                    #     'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                    #     'Loss: {loss.val:.4f} ({loss.avg:.4f})\t' \
                    #     'Prec@1: {top.val:.3f} ({top.avg:.3f})\t'.format(
                    #     batch_idx, len(self.val_loader), epoch=self.epoch, iteration=self.iteration,
                    #     batch_time=batch_time, loss=losses, top=top)
                    log_str = 'Test: [{0}/{1}/{top.count:}]\nepoch: {epoch:}\titer: {iteration:}\t' \
                        'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                        'Loss: {loss.val:} ({loss.avg:})\t' \
                        'Prec@1: {top.val:} ({top.avg:})\t'.format(
                        batch_idx, len(self.val_loader), epoch=self.epoch, iteration=self.iteration,
                        batch_time=batch_time, loss=losses, top=top)
                    print(log_str)
                    self.print_log(log_str)

        else:
            for batch_idx, (imgs1, imgs2, relations, target) in tqdm.tqdm(
                enumerate(self.val_loader), total=len(self.val_loader),
                desc='Valid iteration={} epoch={}'.format(self.iteration, self.epoch), ncols=80, leave=True, position=0):

                gc.collect()
                if self.cuda:
                    imgs1, imgs2, target = imgs1.cuda(), imgs2.cuda(), target.cuda(non_blocking=True)
                imgs1 = Variable(imgs1, volatile=True)
                imgs2 = Variable(imgs2, volatile=True)
                target = Variable(target, volatile=True)
                
                output = self.model(imgs1, imgs2)
                loss = self.criterion(output, target)

                if np.isnan(float(loss.data[0])):
                    raise ValueError('loss is nan while validating')

                # measure accuracy and record loss
                prec = utils.accuracy(output.data, target.data)
                losses.update(loss.data, imgs1.size(0))
                top.update(prec[0], imgs1.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                if batch_idx % self.print_freq == 0:
                    # log_str = 'Test: [{0}/{1}/{top.count:}]\tepoch: {epoch:}\titer: {iteration:}\t' \
                    #     'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                    #     'Loss: {loss.val:.4f} ({loss.avg:.4f})\t' \
                    #     'Prec@1: {top.val:.3f} ({top.avg:.3f})\t'.format(
                    #     batch_idx, len(self.val_loader), epoch=self.epoch, iteration=self.iteration,
                    #     batch_time=batch_time, loss=losses, top=top)
                    log_str = 'Test: [{0}/{1}/{top.count:}]\nepoch: {epoch:}\titer: {iteration:}\t' \
                        'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                        'Loss: {loss.val:} ({loss.avg:})\t' \
                        'Prec@1: {top.val:} ({top.avg:})\t'.format(
                        batch_idx, len(self.val_loader), epoch=self.epoch, iteration=self.iteration,
                        batch_time=batch_time, loss=losses, top=top)
                    print(log_str)
                    self.print_log(log_str)

        self.writer.add_scalar("validation loss", losses.avg, global_step=self.step_val)
        self.writer.add_scalar("training accuracy", top.avg, global_step=self.step_val)
        self.step_val = self.step_val+1
        if self.cmd == 'train':
            is_best = top.avg > self.best_top
            self.best_top = max(top.avg, self.best_top)

            # log_str = 'Test_summary: [{0}/{1}/{top.count:}] epoch: {epoch:} iter: {iteration:}\t' \
            #       'BestPrec@1: {best_top:.3f}\t' \
            #       'Time: {batch_time.avg:.3f}\tLoss: {loss.avg:.4f}\t' \
            #       'Prec@1: {top.avg:.3f}\t'.format(
            #     batch_idx, len(self.val_loader), epoch=self.epoch, iteration=self.iteration,
            #     best_top=self.best_top, batch_time=batch_time, loss=losses, top=top)
            log_str = 'Test_summary: [{0}/{1}/{top.count:}]\nepoch: {epoch:} iter: {iteration:}\t' \
                  'BestPrec@1: {best_top:}\t' \
                  'Time: {batch_time.avg:.3f}\tLoss: {loss.avg:}\t' \
                  'Prec@1: {top.avg:}\t'.format(
                batch_idx, len(self.val_loader), epoch=self.epoch, iteration=self.iteration,
                best_top=self.best_top, batch_time=batch_time, loss=losses, top=top)
            print(log_str)
            self.print_log(log_str)

            if self.dataset_type == "scratch":
                model_state = self.model.state_dict()
            elif self.dataset_type == "vggface2":
                model_state = self.model.fc.state_dict()
            else:
                model_state = self.model.fnet.state_dict()

            checkpoint_file = os.path.join(self.checkpoint_dir, self.checkpoint_file)
            torch.save({
                'epoch': self.epoch,
                'iteration': self.iteration,
                'arch': self.model.__class__.__name__,
                'optim_state_dict': self.optim.state_dict(),
                'model_state_dict': model_state,
                'best_top': self.best_top,
                'batch_time': batch_time,
                'losses': losses,
                'top': top,
            }, checkpoint_file)
            if is_best:
                shutil.copy(checkpoint_file, os.path.join(self.checkpoint_dir, f'model_best_{self.dataset_type}_{self.arch_type}.pth.tar'))
            if (self.epoch + 1) % 10 == 0: # save each 10 epoch
                shutil.copy(checkpoint_file, os.path.join(self.checkpoint_dir, f'checkpoint-{self.dataset_type}-{self.arch_type}-{self.epoch}.pth.tar'))

            if training:
                self.model.train()

    def train_epoch(self):
        batch_time = utils.AverageMeter()
        data_time = utils.AverageMeter()
        losses = utils.AverageMeter()
        top = utils.AverageMeter()

        self.model.train()
        self.optim.zero_grad()

        checkpoint_interval = len(self.train_loader)//4

        end = time.time()
        if self.dataset == "fiw":
            for batch_idx, (imgs1, imgs2, ages1, ages2, genders1, genders2, target) in tqdm.tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                    desc='Train epoch={}, iter={}'.format(self.epoch, self.iteration), ncols=80, leave=True, position=0):
                iteration = batch_idx + self.epoch * len(self.train_loader)
                data_time.update(time.time() - end)

                gc.collect()
                target = target.view(-1).long()
                if self.iteration != 0 and (iteration - 1) != self.iteration:
                    continue  # for resuming
                self.iteration = iteration

                if (self.iteration + 1) % self.interval_validate == 0:    
                    self.validate()

                if self.cuda:
                    imgs1, imgs2, target = imgs1.cuda(), imgs2.cuda(), target.cuda(non_blocking=True)
                imgs1, imgs2, target = Variable(imgs1), Variable(imgs2), Variable(target)

                output = self.model(imgs1, imgs2)
                loss = self.criterion(output, target)
                if np.isnan(float(loss.data)):
                    raise ValueError('loss is nan while training')

                # measure accuracy and record loss
                prec = utils.accuracy(output.data, target.data)
                losses.update(loss.data, imgs1.size(0))
                top.update(prec[0], imgs1.size(0))

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                if self.iteration % self.print_freq == 0:
                    #print(type(top.count), type(batch_time.val),type(data_time.val),type(losses.val),type(top.val))
                    # log_str = 'Train: [{0}/{1}/{top.count:}]\tepoch: {epoch:}\titer: {iteration:}\t' \
                    #     'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                    #     'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                    #     'Loss: {loss.val:.4f} ({loss.avg:.4f})\t' \
                    #     'Prec@1: {top.val:.3f} ({top.avg:.3f})\t'.format(
                    #     batch_idx, len(self.train_loader), epoch=self.epoch, iteration=self.iteration,
                    #     lr=self.optim.param_groups[0]['lr'],
                    #     batch_time=batch_time, data_time=data_time, loss=losses, top=top)

                    log_str = 'Train: [{0}/{1}/{top.count:}]\nepoch: {epoch:}\titer: {iteration:}\t' \
                        'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                        'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                        'Loss: {loss.val:} ({loss.avg:})\t' \
                        'Prec@1: {top.val:} ({top.avg:})\t'.format(
                        batch_idx, len(self.train_loader), epoch=self.epoch, iteration=self.iteration,
                        lr=self.optim.param_groups[0]['lr'],
                        batch_time=batch_time, data_time=data_time, loss=losses, top=top)
                    print(log_str)
                    self.print_log(log_str)

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()  # update lr

                #if (batch_idx+1) % 500 == 0:
                if (batch_idx+1) % checkpoint_interval == 0:
                    is_best = top.avg > self.best_top
                    self.best_top = max(top.avg, self.best_top)

                    if self.dataset_type == "scratch":
                        model_state = self.model.state_dict()
                    elif self.dataset_type == "vggface2":
                        model_state = self.model.fc.state_dict()
                    else:
                        model_state = self.model.fnet.state_dict()

                    checkpoint_file = os.path.join(self.checkpoint_dir, self.checkpoint_file)
                    torch.save({
                        'epoch': self.epoch,
                        'iteration': self.iteration,
                        'arch': self.model.__class__.__name__,
                        'optim_state_dict': self.optim.state_dict(),
                        'model_state_dict': model_state,
                        'best_top': self.best_top,
                        'batch_time': batch_time,
                        'losses': losses,
                        'top': top,
                    }, checkpoint_file)
                    if is_best:
                        shutil.copy(checkpoint_file, os.path.join(self.checkpoint_dir, f'model_best_{self.dataset_type}_{self.arch_type}.pth.tar'))
                    if (self.epoch + 1) % 10 == 0: # save each 10 epoch
                        shutil.copy(checkpoint_file, os.path.join(self.checkpoint_dir, f'checkpoint-{self.dataset_type}-{self.arch_type}-{self.epoch}.pth.tar'))

        else:
            for batch_idx, (imgs1, imgs2, relations, target) in tqdm.tqdm(
                    enumerate(self.train_loader), total=len(self.train_loader),
                    desc='Train epoch={}, iter={}'.format(self.epoch, self.iteration), ncols=80, leave=True, position=0):
                iteration = batch_idx + self.epoch * len(self.train_loader)
                data_time.update(time.time() - end)

                gc.collect()

                if self.iteration != 0 and (iteration - 1) != self.iteration:
                    continue  # for resuming
                self.iteration = iteration

                if (self.iteration + 1) % self.interval_validate == 0:
                    self.validate()

                if self.cuda:
                    imgs1, imgs2, target = imgs1.cuda(), imgs2.cuda(), target.cuda(non_blocking=True)
                imgs1, imgs2, target = Variable(imgs1), Variable(imgs2), Variable(target)

                output = self.model(imgs1, imgs2)
                loss = self.criterion(output, target)
                if np.isnan(float(loss.data[0])):
                    raise ValueError('loss is nan while training')

                # measure accuracy and record loss
                prec = utils.accuracy(output.data, target.data)
                losses.update(loss.data, imgs1.size(0))
                top.update(prec[0], imgs1.size(0))

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                if self.iteration % self.print_freq == 0:
                    # log_str = 'Train: [{0}/{1}/{top.count:}]\tepoch: {epoch:}\titer: {iteration:}\t' \
                    #     'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                    #     'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                    #     'Loss: {loss.val:.4f} ({loss.avg:.4f})\t' \
                    #     'Prec@1: {top.val:.3f} ({top.avg:.3f})\t'.format(
                    #     batch_idx, len(self.train_loader), epoch=self.epoch, iteration=self.iteration,
                    #     lr=self.optim.param_groups[0]['lr'],
                    #     batch_time=batch_time, data_time=data_time, loss=losses, top=top)
                    log_str = 'Train: [{0}/{1}/{top.count:}]\nepoch: {epoch:}\titer: {iteration:}\t' \
                        'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                        'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                        'Loss: {loss.val:} ({loss.avg:})\t' \
                        'Prec@1: {top.val:} ({top.avg:})\t'.format(
                        batch_idx, len(self.train_loader), epoch=self.epoch, iteration=self.iteration,
                        lr=self.optim.param_groups[0]['lr'],
                        batch_time=batch_time, data_time=data_time, loss=losses, top=top)
                    print(log_str)
                    self.print_log(log_str)

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()  # update lr

                if (batch_idx+1) % checkpoint_interval == 0:
                    is_best = top.avg > self.best_top
                    self.best_top = max(top.avg, self.best_top)

                    if self.dataset_type == "scratch":
                        model_state = self.model.state_dict()
                    elif self.dataset_type == "vggface2":
                        model_state = self.model.fc.state_dict()
                    else:
                        model_state = self.model.fnet.state_dict()

                    checkpoint_file = os.path.join(self.checkpoint_dir, self.checkpoint_file)
                    torch.save({
                        'epoch': self.epoch,
                        'iteration': self.iteration,
                        'arch': self.model.__class__.__name__,
                        'optim_state_dict': self.optim.state_dict(),
                        'model_state_dict': model_state,
                        'best_top': self.best_top,
                        'batch_time': batch_time,
                        'losses': losses,
                        'top': top,
                    }, checkpoint_file)
                    if is_best:
                        shutil.copy(checkpoint_file, os.path.join(self.checkpoint_dir, f'model_best_{self.dataset_type}_{self.arch_type}.pth.tar'))
                    if (self.epoch + 1) % 10 == 0: # save each 10 epoch
                        shutil.copy(checkpoint_file, os.path.join(self.checkpoint_dir, f'checkpoint-{self.dataset_type}-{self.arch_type}-{self.epoch}.pth.tar'))        
        

        # log_str = 'Train_summary: [{0}/{1}/{top.count:}]\nepoch: {epoch:}\titer: {iteration:}\t' \
        #               'Time: {batch_time.avg:.3f}\tData: {data_time.avg:.3f}\t' \
        #               'Loss: {loss.avg:.4f}\tPrec@1: {top.avg:.3f}\t'.format(
        #             batch_idx, len(self.train_loader), epoch=self.epoch, iteration=self.iteration,
        #             lr=self.optim.param_groups[0]['lr'],
        #             batch_time=batch_time, data_time=data_time, loss=losses, top=top)
        log_str = 'Train_summary: [{0}/{1}/{top.count:}]\nepoch: {epoch:}\titer: {iteration:}\t' \
                      'Time: {batch_time.avg:.3f}\tData: {data_time.avg:.3f}\t' \
                      'Loss: {loss.avg:}\tPrec@1: {top.avg:}\t'.format(
                    batch_idx, len(self.train_loader), epoch=self.epoch, iteration=self.iteration,
                    lr=self.optim.param_groups[0]['lr'],
                    batch_time=batch_time, data_time=data_time, loss=losses, top=top)
        print(log_str)
        self.print_log(log_str)
        self.writer.add_scalar("training loss", losses.avg, global_step=self.step)
        self.writer.add_scalar("training accuracy", top.avg, global_step=self.step)
        self.step = self.step + 1

    def test(self):
        print("testing...")
        num_correct = 0
        num_samples = 0
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (imgs1, imgs2,target) in enumerate(
                DataLoader(self.test_data, sampler=RandomSampler(self.test_data, replacement=True, num_samples=128), batch_size=16, drop_last=True)):
                target = target.view(-1).long()
                if self.cuda:
                    imgs1 = imgs1.cuda()
                    imgs2 = imgs2.cuda()
                    target = target.cuda()

                scores = self.model(imgs1, imgs2)
                _, predictions = scores.max(1)
                num_correct += (predictions == target).sum()
                num_samples += predictions.size(0)

        acc = (float(num_correct)/float(num_samples)*100)
        s = f"\nnum_correct: {num_correct}\tnum_samples: {num_samples}\tTesting accuracy: {acc}"
        print(s)
        self.print_log(s)
        self.writer.add_scalar("testing accuracy", acc, self.step_test)
        self.step_test = self.step_test + 1

    def train(self):
        #max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader))) # 117
        max_epoch = 100
        self.writer = SummaryWriter(self.tb_dir)
        for epoch in tqdm.trange(self.epoch, max_epoch, desc='Train', ncols=80):
            self.epoch = epoch
            self.train_epoch()
            if self.epoch+1 % 2 == 0:
                self.test()
            if self.iteration >= self.max_iter:
                break




class Validator(Trainer):

    def __init__(self, cmd, cuda, model, criterion, val_loader, log_file, print_freq=1):
        super(Validator, self).__init__(cmd, cuda=cuda, model=model, criterion=criterion,
                                        val_loader=val_loader, log_file=log_file, print_freq=print_freq,
                                        optimizer=None, train_loader=None, max_iter=None,
                                        interval_validate=None, lr_scheduler=None,
                                        checkpoint_dir=None)

    def train(self):
        raise NotImplementedError