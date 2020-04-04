import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from utils.plot_grad_flow import plot_grad_flow

class EarlyStopping:
    def __init__(self, checkpoint_name, lr_scheduler, patiences=[], delta=0):
        self.checkpoint_name = checkpoint_name
        self.lr_scheduler = lr_scheduler
        self.ignore_times = len(patiences)
        self.patience_idx = 0
        self.patiences = patiences
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1

            if self.counter >= self.patiences[self.patience_idx]:
                if self.patience_idx < self.ignore_times - 1:
                    model.load_state_dict(torch.load(self.checkpoint_name))
                    self.lr_scheduler.step()
                    self.patience_idx += 1
                    self.counter = 0
                else:
                    self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.checkpoint_name)
        self.val_loss_min = val_loss

def train(train_loader,model,loss_fn,optimizer,epoch):
    model.train()
    losses=[]
    for batch_idx, (samples,labels) in enumerate(train_loader):
        if len(samples) == 2:
            feature_1 = samples[0].float()
            feature_2 = samples[1].float()
            if torch.cuda.is_available():
                feature_1 = feature_1.cuda()
                feature_2 = feature_2.cuda()
                labels = labels.cuda()
            feature_1 = Variable(feature_1)
            feature_2 = Variable(feature_2)
            labels = Variable(labels)
            predicted_value = model(feature_1, feature_2)
        else:
            features = samples.float()
            if torch.cuda.is_available():
                features = features.cuda()
                labels = labels.cuda()
            features = Variable(features)
            labels = Variable(labels)
            predicted_value = model(features)
            
        loss = loss_fn(predicted_value,labels)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        # plot_grad_flow(model.named_parameters())
        # nn.utils.clip_grad_norm_(model.parameters(), 0.2)
        optimizer.step()
        # if batch_idx % 10 == 0:
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, batch_idx * len(samples), len(train_loader.dataset),
            #     100. * batch_idx / len(train_loader), loss.item()))

    return losses

def test(test_loader,model,loss_fn):
    model.eval()
    losses = []
    correct=0
    with torch.no_grad():
        for batch_idx, (samples,labels) in enumerate(test_loader):
            if len(samples) == 2:
                feature_1 = samples[0].float()
                feature_2 = samples[1].float()
                if torch.cuda.is_available():
                    feature_1 = feature_1.cuda()
                    feature_2 = feature_2.cuda()
                    labels = labels.cuda()
                predicted_value = model(feature_1, feature_2)
            else:
                features=samples.float()
                if torch.cuda.is_available():
                    features= features.cuda()
                    labels=labels.cuda()

                predicted_value=model(features)

            loss=loss_fn(predicted_value,labels)
            _, predicted_value=torch.max(predicted_value,1)
            correct_number=(predicted_value==labels).sum()

            losses.append(loss.item())
            correct+=correct_number.item()
    
    return losses, correct 
