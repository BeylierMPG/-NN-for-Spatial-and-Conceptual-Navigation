import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data.dataset import random_split
from IPython.display import clear_output

from matplotlib import pyplot as plt


import torch.nn as nn
import time





class Net_Multi(torch.nn.Module):
    def __init__(self,input_dimension,nodes_second,nodes_third,nodes_output):
        super(Net_Multi, self).__init__()

        
        
        self.fc1 = nn.Sequential(
            nn.Linear(in_features = input_dimension, out_features = nodes_second),    
            nn.ReLU()
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(in_features = nodes_second, out_features = nodes_third),    
            nn.ReLU()
        )
        
        self.fc3 = nn.Sequential(
            nn.Linear(in_features = nodes_third,  out_features = nodes_output)
        )
        
        
        #self.fc1 = nn.Linear(in_features = input_dimension, out_features = nodes_second)
        #self.fc2 = nn.Linear(in_features = nodes_second, out_features = nodes_third)
        #self.fc3 = nn.Linear(in_features = nodes_third,  out_features = nodes_output)

        #self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(nodes_second)
        self.batchnorm2 = nn.BatchNorm1d(nodes_third)
        

        self.mask_sub = torch.tensor( [[1 if i<nodes_third/2 else 0 for i in range(nodes_third)]if k<1
                                   else [0 if i<nodes_third/2 else 1 for i in range(nodes_third)]
                                   for k in range(nodes_output)],
                                 dtype= torch.float64)

        with torch.no_grad():
            self.fc3[0].weight.mul_(self.mask_sub)
        

    def forward(self, x):
        #x = self.relu(self.fc1(x))
        x = self.fc1(x)
        x = self.batchnorm1(x)
        #x = self.relu(self.fc2(x))
        x = self.fc2(x)
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
    
    

    
    
class Net_Individual(torch.nn.Module):
    def __init__(self,input_dimension,nodes_second,nodes_third,nodes_output):
        super(Net_Individual, self).__init__()
        
        
        
        self.fc1 = nn.Sequential(
            nn.Linear(in_features = input_dimension, out_features = nodes_second),    
            nn.ReLU()
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(in_features = nodes_second, out_features = nodes_third),    
            nn.ReLU()
        )
        
        self.fc3 = nn.Sequential(
            nn.Linear(in_features = nodes_third,  out_features = nodes_output)
        )
        

    
    
        #self.fc1 = nn.Linear(in_features = input_dimension, out_features = nodes_second)
        #self.fc2 = nn.Linear(in_features = nodes_second, out_features = nodes_third)
        #self.fc3 = nn.Linear(in_features = nodes_third,  out_features = nodes_output)

        #self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(nodes_second)
        self.batchnorm2 = nn.BatchNorm1d(nodes_third)

        
       

    def forward(self, x):
        
        #x = self.relu(self.fc1(x))
        x = self.fc1(x)
        x = self.batchnorm1(x)
        #x = self.relu(self.fc2(x))
        x = self.fc2(x)
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
    
    

    
    
class Training():
    
    def __init__(self):
        pass
    
    def training_individual(self,Epoch,train_loader,val_loader):
        
        model = Net_Individual(input_dimension=5,nodes_second = 20,nodes_third = 5,nodes_output = 1)    
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        criterion = nn.BCEWithLogitsLoss()


        losses = []
        val_losses = []

        for epoch in range(Epoch):

            loss = 0
            val_loss = 0 

            for x_batch, y_batch in train_loader:

                x_batch = x_batch[:,0:5]

                optimizer.zero_grad()
                output = model(x_batch).squeeze(1)

                output_loss = criterion(output,y_batch)
                output_loss.backward()
                optimizer.step()

                loss += output_loss.detach().numpy()
            losses.append(loss/len(train_loader))

            with torch.no_grad():
                for x_val, y_val in val_loader:
                    model.eval() 
                    x_val = x_val[:,0:5]
                    yhat = model(x_val).squeeze(1)
                    val_loss += criterion(yhat,y_val)
                val_losses.append(val_loss.item()/len(val_loader))


            if np.mod(epoch,10)==0: 
                clear_output()
                print("Epoch {}, val_loss {}".format(epoch, val_loss.item()/len(val_loader))) 
                fig = plt.figure()
                ax = fig.add_subplot(111)
                plt.plot(np.arange(len(val_losses)), val_losses)
                plt.ylabel('Loss')
                plt.xlabel('Epoch #')
                plt.show()
       
        return model

    def training_multi(self,Epoch,train_loader,val_loader):
        
        model = Net_Multi(input_dimension=6,nodes_second = 3,nodes_third = 10,nodes_output = 2)    
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        criterion = nn.BCEWithLogitsLoss()
        mask = Masks(model)

        
        losses = []
        val_losses = []
        
        for epoch in range(Epoch):

            loss = 0
            val_loss = 0 

            for x_batch, y_batch in train_loader:


                pumpkin_seed= x_batch[0,5]
                #x_batch = x_batch[:,0:5].squeeze(0)
                x_batch = x_batch.squeeze(0)

                optimizer.zero_grad()

                output = model(x_batch)

                output_loss = criterion(output,y_batch.squeeze(0))
                output_loss.backward()

                mask.apply_mask(model,pumpkin_seed)                    
                optimizer.step()

                loss += output_loss.detach().numpy()
            losses.append(loss/len(train_loader))

            with torch.no_grad():
                for x_val, y_val in val_loader:
                    model.eval() # not useful for now
                    #x_val = x_val[:,0:5].squeeze(0)
                    x_val = x_val.squeeze(0)
                    yhat = model(x_val)
                    val_loss += criterion(yhat,y_val.squeeze(0))
                val_losses.append(val_loss.item()/len(val_loader))


            if np.mod(epoch,10)==0: 
                clear_output()
                print("Epoch {}, val_loss {}".format(epoch, val_loss.item()/len(val_loader))) 
                fig = plt.figure()
                ax = fig.add_subplot(111)
                plt.plot(np.arange(len(val_losses)), val_losses)
                plt.ylabel('Loss')
                plt.xlabel('Epoch #')
                plt.show()
                
       
        return model
            






            


    

class Masks():
    
    def __init__(self,model):
        nodes_second = model.fc2[0].weight.shape[1]
        nodes_third = model.fc3[0].weight.shape[1]
        nodes_output = model.fc3[0].weight.shape[0]

        self.mask_conceptual_3 = torch.tensor( [[1 if i<nodes_third/2 else 0 for i in range(nodes_third)] if k<1
                                        else [0 for i in range(nodes_third)]
                                        for k in range(nodes_output)],
                                      dtype= torch.float64)
              
              
        self.mask_spatial_3 = torch.tensor( [[ 0 for i in range(nodes_third)] if k<1
                                        else [0 if i<nodes_third/2 else 1 for i in range(nodes_third)]
                                        for k in range(nodes_output)],
                                      dtype= torch.float64)
        
        self.mask_conceptual_2 = torch.tensor( [[1  for i in range(nodes_second)] if k<nodes_third/2
                                        else [0 for i in range(nodes_second)]
                                        for k in range(nodes_third)],
                                      dtype= torch.float64)
              
              
        self.mask_spatial_2 = torch.tensor( [[ 0 for i in range(nodes_second)] if k<nodes_third/2
                                        else [1 for i in range(nodes_second)]
                                        for k in range(nodes_third)],
                                      dtype= torch.float64)
        
        
    

    def apply_mask(self,model,pumpkin_seed):
        with torch.no_grad():
            if pumpkin_seed == 1:
                model.fc3[0].weight.grad.mul_(self.mask_conceptual_3)
                model.fc2[0].weight.grad.mul_(self.mask_conceptual_2)
            if pumpkin_seed == 0:
                model.fc3[0].weight.grad.mul_(self.mask_spatial_3)
                model.fc2[0].weight.grad.mul_(self.mask_spatial_2)
                
                
                
class Activation_Hook():
    
    def __init__(self):
        self.activation = {}
        
        
    def getActivation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook
            
    
    def registration(self,model):
        
        self.h1 = model.fc1.register_forward_hook(self.getActivation('fc1'))
        self.h2 = model.fc2.register_forward_hook(self.getActivation('fc2'))
        self.h3 = model.fc3.register_forward_hook(self.getActivation('fc3'))
        
        

    def detach(self):
        self.h1.remove()
        self.h2.remove()
        self.h3.remove()
 
        
