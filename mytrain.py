import os
import torch 
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn


def get_batch_size():
    batch_size = 
    return batch_size
    
    
def get_learning_rate():
    learning_rate =
    return learning_rate


def get_num_epoch():
    num_epoch = 
    return num_epoch


def compute_accuracy(pred, label):
    accuracy =  
    return accuracy

    

class Trainer:
    def __init__(self,
        model,
        num_class,
    ):
        self.model      = model
        self.optimizer  = optim.SGD(params=model.parameters(), lr=get_learning_rate())
        self.criterion  = 
        self.num_class  = num_class


    def train(self, data, label):
        self.model.train()
        # ==========
        
        # ==========
        accuracy        = compute_accuracy(pred, label) 
        loss_val        = loss.item()
        accuracy_val    = accuracy.item()
        
        return (loss_val, accuracy_val) 
       
        
    def eval(self, data, label):
        self.model.eval()
        # ==========
        
        # ==========
        accuracy        = compute_accuracy(pred, label)
        loss_val        = loss.item()
        accuracy_val    = accuracy.item()
        
        return (loss_val, accuracy_val) 
       
    
    def get_one_hot_encoding(self, label):
        target = torch.zeros(len(label), self.num_class) 
        # ==========
        
        # ==========
        return target 