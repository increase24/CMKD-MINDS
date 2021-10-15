import torch
from .stats import AverageMeter, accuracy
import numpy as np

class Trainer(object):
    def __init__(self, train_loader, valid_loader, model, device, criterion, optimizer, print_freq):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.print_freq = print_freq
        self.bst_acc = 0.0
        self.flag_improve = False

    def reset_optimiser(self, optimizer):
        self.optimizer = optimizer

    def train_epoch(self, epoch):
        losses = AverageMeter()
        accuracies = AverageMeter()
        self.model.train()
        for idx, (input, target) in enumerate(self.train_loader):
            # compute output and loss
            input, target = input.to(self.device), target.to(self.device)
            self.model.zero_grad()
            output = self.model(input)
            loss = self.criterion(output, target)
            losses.update(loss.item(), input.size(0))
            # measure accuracy
            [acc] = accuracy(output.detach(), target.detach().cpu())
            accuracies.update(acc.item(), input.size(0))
            # compute grandient and do back propagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if idx % self.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc: {acc.val:.4f} ({acc.avg:.4f})'.format(
                        epoch, idx, len(self.train_loader),
                        loss = losses, acc = accuracies 
                    ))
        print('Epoch: [{0}][{1}/{2}]\t'
                'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                'Acc: {acc.val:.4f} ({acc.avg:.4f})'.format(
                    epoch, idx, len(self.train_loader),
                    loss = losses, acc = accuracies 
                ))
        return losses.avg, accuracies.avg
                
                
    def validate(self, eval_only = False):
        losses = AverageMeter()
        accuracies = AverageMeter()
        self.model.eval()
        with torch.no_grad():
            for idx, (input, target) in enumerate(self.valid_loader):
                input, target = input.to(self.device), target.to(self.device)
                output = self.model(input)
                loss = self.criterion(output, target)
                [acc] = accuracy(output.detach(), target.detach().cpu())
                losses.update(loss.item(), input.size(0))
                accuracies.update(acc.item(), input.size(0))
                if idx % self.print_freq == 0:
                    print(
                        'Test: [{0}/{1}]\t'
                        'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Acc: {acc.val:.4f} ({acc.avg:.4f})'.format(
                        idx, len(self.valid_loader),
                        loss = losses, acc = accuracies 
                    ))
            print('Test: [{0}/{1}]\t'
                'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                'Acc: {acc.val:.4f} ({acc.avg:.4f})'.format(
                idx, len(self.valid_loader),
                loss = losses, acc = accuracies 
            ))
            if(accuracies.avg>self.bst_acc):
                self.bst_acc = accuracies.avg
                self.flag_improve = True
            else:
                self.flag_improve = False
        return losses.avg, accuracies.avg

class Trainer_kd(object):
    def __init__(self, train_loader, valid_loader, model_stu, model_tch, device, criterion, optimizer, print_freq):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model_stu = model_stu
        self.model_tch = model_tch
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.print_freq = print_freq
        self.bst_acc = 0.0
        self.flag_improve = False

    def train_kd_epoch(self, epoch):
        losses = AverageMeter()
        accuracies = AverageMeter()
        self.model_stu.train()
        self.model_tch.eval()
        for idx, (input_stu, input_tch, target) in enumerate(self.train_loader):
            input_stu, input_tch, target =  input_stu.to(self.device), input_tch.to(self.device), target.to(self.device)
            self.model_stu.zero_grad()
            output_stu = self.model_stu(input_stu)
            output_tch = self.model_tch(input_tch)
            loss = self.criterion(output_stu, target, output_tch)
            losses.update(loss.item(), input_stu.size(0))
            # measure accuracy
            [acc] = accuracy(output_stu.detach(), target.detach().cpu())
            accuracies.update(acc.item(), input_stu.size(0))
            # compute grandient and do back propagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if idx % self.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc: {acc.val:.4f} ({acc.avg:.4f})'.format(
                        epoch, idx, len(self.train_loader),
                        loss = losses, acc = accuracies 
                    ))
        print('Epoch: [{0}][{1}/{2}]\t'
                'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                'Acc: {acc.val:.4f} ({acc.avg:.4f})'.format(
                    epoch, idx, len(self.train_loader),
                    loss = losses, acc = accuracies 
                ))
        return losses.avg, accuracies.avg

    def validate_kd(self, eval_only = False):
        losses = AverageMeter()
        accuracies = AverageMeter()
        self.model_stu.eval()
        self.model_tch.eval()
        with torch.no_grad():
            for idx, (input_stu, input_tch, target) in enumerate(self.valid_loader):
                input_stu, input_tch, target = input_stu.to(self.device), input_tch.to(self.device), target.to(self.device)
                output_stu = self.model_stu(input_stu)
                output_tch = self.model_tch(input_tch)
                loss = self.criterion(output_stu, target, output_tch)
                [acc] = accuracy(output_stu.detach(), target.detach().cpu())
                losses.update(loss.item(), input_stu.size(0))
                accuracies.update(acc.item(), input_stu.size(0))
                if idx % self.print_freq == 0:
                    print(
                        'Test: [{0}/{1}]\t'
                        'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Acc: {acc.val:.4f} ({acc.avg:.4f})'.format(
                        idx, len(self.valid_loader),
                        loss = losses, acc = accuracies 
                    ))
            print('Test: [{0}/{1}]\t'
                'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                'Acc: {acc.val:.4f} ({acc.avg:.4f})'.format(
                idx, len(self.valid_loader),
                loss = losses, acc = accuracies 
            ))
            if(accuracies.avg>self.bst_acc):
                self.bst_acc = accuracies.avg
                self.flag_improve = True
            else:
                self.flag_improve = False
        return losses.avg, accuracies.avg


class Trainer_Bi(object):
    def __init__(self, train_loader, valid_loader, model, device, criterion, optimizer, print_freq):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.print_freq = print_freq
        self.bst_acc_positive = 0.0
        self.flag_improve_positive = False
        self.bst_acc_reverse = 0.0
        self.flag_improve_reverse = False
        self.bst_acc = 0.0
        self.flag_improve = False

    def train_Bi_stage1_epoch(self, epoch):
        losses_positive = AverageMeter()
        accuracies_positive = AverageMeter()
        losses_reverse = AverageMeter()
        accuracies_reverse = AverageMeter()
        self.model.train()
        for idx, (input, target) in enumerate(self.train_loader):
            input, target =  input.to(self.device), target.to(self.device)
            
            # positive direction 
            self.model.zero_grad()
            output_positive = self.model(input, 'stage1-positive')
            loss = self.criterion(output_positive, target)
            losses_positive.update(loss.item(), input.size(0))
            ## measure accuracy
            [acc] = accuracy(output_positive.detach(), target.detach().cpu())
            accuracies_positive.update(acc.item(), input.size(0))
            ## compute grandient and do back propagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # reverse direction
            self.model.zero_grad()
            output_reverse = self.model(input, 'stage1-reverse')
            loss = self.criterion(output_reverse, target)
            losses_reverse.update(loss.item(), input.size(0))
            ## measure accuracy
            [acc] = accuracy(output_reverse.detach(), target.detach().cpu())
            accuracies_reverse.update(acc.item(), input.size(0))
            ## compute grandient and do back propagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if idx % self.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Loss: {loss_p.val:.4f} ({loss_p.avg:.4f})\t'
                    'Acc: {acc_p.val:.4f} ({acc_p.avg:.4f})'
                    'Loss: {loss_r.val:.4f} ({loss_r.avg:.4f})\t'
                    'Acc: {acc_r.val:.4f} ({acc_r.avg:.4f})'.format(
                        epoch, idx, len(self.train_loader),
                        loss_p = losses_positive, acc_p = accuracies_positive,
                        loss_r = losses_reverse, acc_r = accuracies_reverse
                    ))
        print('Epoch: [{0}][{1}/{2}]\t'
                'Loss: {loss_p.val:.4f} ({loss_p.avg:.4f})\t'
                'Acc: {acc_p.val:.4f} ({acc_p.avg:.4f})'
                'Loss: {loss_r.val:.4f} ({loss_r.avg:.4f})\t'
                'Acc: {acc_r.val:.4f} ({acc_r.avg:.4f})'.format(
                    epoch, idx, len(self.train_loader),
                    loss_p = losses_positive, acc_p = accuracies_positive,
                    loss_r = losses_reverse, acc_r = accuracies_reverse
                ))
        return losses_positive.avg, accuracies_positive.avg, losses_reverse.avg, accuracies_reverse.avg

    def validate_Bi_stage1(self, eval_only = False):
        losses_positive = AverageMeter()
        accuracies_positive = AverageMeter()
        losses_reverse = AverageMeter()
        accuracies_reverse = AverageMeter()
        self.model.eval()
        with torch.no_grad():
            for idx, (input, target) in enumerate(self.valid_loader):
                input, target = input.to(self.device), target.to(self.device)
                # positive direction 
                output_positive = self.model(input, 'stage1-positive')
                loss = self.criterion(output_positive, target)
                [acc] = accuracy(output_positive.detach(), target.detach().cpu())
                losses_positive.update(loss.item(), input.size(0))
                accuracies_positive.update(acc.item(), input.size(0))
                # reverse direction 
                output_reverse = self.model(input, 'stage1-reverse')
                loss = self.criterion(output_reverse, target)
                [acc] = accuracy(output_reverse.detach(), target.detach().cpu())
                losses_reverse.update(loss.item(), input.size(0))
                accuracies_reverse.update(acc.item(), input.size(0))
                if idx % self.print_freq == 0:
                    print(
                        'Test: [{0}/{1}]\t'
                        'Loss: {loss_p.val:.4f} ({loss_p.avg:.4f})\t'
                        'Acc: {acc_p.val:.4f} ({acc_p.avg:.4f})'
                        'Loss: {loss_r.val:.4f} ({loss_r.avg:.4f})\t'
                        'Acc: {acc_r.val:.4f} ({acc_r.avg:.4f})'.format(
                        idx, len(self.valid_loader),
                        loss_p = losses_positive, acc_p = accuracies_positive,
                        loss_r = losses_reverse, acc_r = accuracies_reverse
                    ))
            print('Test: [{0}/{1}]\t'
                'Loss: {loss_p.val:.4f} ({loss_p.avg:.4f})\t'
                'Acc: {acc_p.val:.4f} ({acc_p.avg:.4f})'
                'Loss: {loss_r.val:.4f} ({loss_r.avg:.4f})\t'
                'Acc: {acc_r.val:.4f} ({acc_r.avg:.4f})'.format(
                idx, len(self.valid_loader),
                loss_p = losses_positive, acc_p = accuracies_positive,
                loss_r = losses_reverse, acc_r = accuracies_reverse
            ))
            if(accuracies_positive.avg>self.bst_acc_positive):
                self.bst_acc_positive = accuracies_positive.avg
                self.flag_improve_positive = True
            else:
                self.flag_improve_positive = False
            if(accuracies_reverse.avg>self.bst_acc_reverse):
                self.bst_acc_reverse = accuracies_reverse.avg
                self.flag_improve_reverse = True
            else:
                self.flag_improve_reverse = False
        return losses_positive.avg, accuracies_positive.avg, losses_reverse.avg, accuracies_reverse.avg

    def train_Bi_stage2_epoch(self, epoch):
        losses = AverageMeter()
        accuracies = AverageMeter()
        self.model.train()
        for idx, (input, target) in enumerate(self.train_loader):
            input, target =  input.to(self.device), target.to(self.device)
            
            # positive direction 
            self.model.zero_grad()
            output = self.model(input, 'stage2')
            loss = self.criterion(output, target)
            losses.update(loss.item(), input.size(0))
            ## measure accuracy
            [acc] = accuracy(output.detach(), target.detach().cpu())
            accuracies.update(acc.item(), input.size(0))
            ## compute grandient and do back propagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if idx % self.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc: {acc.val:.4f} ({acc.avg:.4f})'.format(
                        epoch, idx, len(self.train_loader),
                        loss = losses, acc = accuracies
                    ))
        print('Epoch: [{0}][{1}/{2}]\t'
                    'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc: {acc.val:.4f} ({acc.avg:.4f})'.format(
                        epoch, idx, len(self.train_loader),
                        loss = losses, acc = accuracies
                    ))
        return losses.avg, accuracies.avg

    def validate_Bi_stage2(self, eval_only = False):
        losses = AverageMeter()
        accuracies = AverageMeter()
        self.model.eval()
        with torch.no_grad():
            for idx, (input, target) in enumerate(self.valid_loader):
                input, target = input.to(self.device), target.to(self.device)
                output = self.model(input, 'stage2')
                loss = self.criterion(output, target)
                [acc] = accuracy(output.detach(), target.detach().cpu())
                losses.update(loss.item(), input.size(0))
                accuracies.update(acc.item(), input.size(0))

                if idx % self.print_freq == 0:
                    print(
                        'Test: [{0}/{1}]\t'
                        'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Acc: {acc.val:.4f} ({acc.avg:.4f})'.format(
                        idx, len(self.train_loader),
                        loss = losses, acc = accuracies
                    ))
            print('Test: [{0}/{1}]\t'
                'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc: {acc.val:.4f} ({acc.avg:.4f})'.format(
                        idx, len(self.train_loader),
                        loss = losses, acc = accuracies
                ))
            if(accuracies.avg>self.bst_acc):
                self.bst_acc = accuracies.avg
                self.flag_improve = True
            else:
                self.flag_improve = False
        return losses.avg, accuracies.avg