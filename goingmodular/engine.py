
"""
Contains function for training and testing a Pytorch model
"""
from tqdm.auto import tqdm
import torch
from typing import Dict,List,Tuple
device = "cuda" if torch.cuda.is_available() else "cpu"
from torch import nn
#the main function is "train" because it have the other train_step and test_step method and before use train function define dataloaders,loss and optimizer functions
def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device = device):
    model.train()
    train_loss, train_acc = 0, 0
     
    for batch,(image,label) in enumerate(data_loader):
        
        #you must be sure about model and data in same device
        image,label=image.to(device),label.to(device)
        
        #take the logits of prediction for full batch 
        y_pred=model(image)
        
        #calculate the loss
        loss=loss_fn(y_pred,label)
        
        #add total loss variable
        train_loss+=loss.item()
        
        #optimizer should be zero before new iteration
        optimizer.zero_grad()
        
        #calculate which weight will change and how much
        loss.backward()
        
        #implement back propagation
        optimizer.step()
        
        #take the logits implement softmax for taking probablities and argmax for take prediction label 
        y_pred_class=torch.argmax(torch.softmax(y_pred,dim=1),dim=1)
        
        #calculate the accuracy
        train_acc +=(y_pred_class==label).sum().item()/len(y_pred)
    
    #devide the loss and accuracy with image number for taking right percent
    train_loss /= len(data_loader)
    train_acc /= len(data_loader) 
    return train_loss,train_acc

def test_step(model:torch.nn.Module,
             data_loader: torch.utils.data.DataLoader,
              loss_fn:torch.nn.Module,
              device:torch.device =device): 
    test_loss,test_acc=0,0
    
    #we should stop calculating gradient with model.eval for testing
    model.eval()
    with torch.inference_mode():
        for batch,(image,label) in enumerate(data_loader):
            image,label=image.to(device),label.to(device)
            
            test_pred_logits=model(image)
            
            loss=loss_fn(test_pred_logits,label)
            test_loss += loss.item()
            
            test_pred_labels=test_pred_logits.argmax(dim=1)
            
            test_acc+= ((test_pred_labels==label).sum().item()/len(test_pred_labels))
            
        test_loss/=len(data_loader)
        test_acc/=len(data_loader)
    return test_loss,test_acc   

def train(model:torch.nn.Module,
          train_dataloader:torch.utils.data.DataLoader,
          test_dataloader:torch.utils.data.DataLoader,
          optimizer:torch.optim.Optimizer,
          loss_fn:torch.nn.Module=nn.CrossEntropyLoss(),
          epochs:int=5,
          device=device):
    
    #we want to put together all results
    results={"train_loss":[],
             "train_acc":[],
             "test_loss":[],
             "test_acc":[]}
    for epoch in tqdm(range(epochs)):
        train_loss,train_acc=train_step(model=model,
                                       data_loader=train_dataloader,
                                       loss_fn=loss_fn,
                                       optimizer=optimizer,
                                       device=device)
        test_loss,test_acc=test_step(model=model,
                                       data_loader=test_dataloader,
                                       loss_fn=loss_fn,
                                       device=device)
        print(f"Epoch:{epoch}| train_loss:{train_loss:.4f}|train_acc:{train_acc:.4f}|test_loss:{test_loss:.4f}|test_acc:{test_acc:.4f}")
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
    return results
