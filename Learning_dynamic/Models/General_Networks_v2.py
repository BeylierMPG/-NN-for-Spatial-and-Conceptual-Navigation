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
import sys
sys.path.append("/Users/charlottebeylier/Documents/PhD/Spatial and Conceptual Learning/github_code/Mono_Multi_TASK")
from Generate_data_activity_spatial_conceptual import Representation_spatial_conceptual



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
        x = self.fc1(x)
        x = self.batchnorm1(x)
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

        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(nodes_second)
        self.batchnorm2 = nn.BatchNorm1d(nodes_third)

        
       

    def forward(self, x):
        
        x = self.fc1(x)
        x = self.batchnorm1(x)
        x = self.fc2(x)
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
    
    

    
    
class Training():
    
    def __init__(self):
        pass
    
    def training_individual(self,Input_Dimension,Nodes_Second,Nodes_Third,Epoch,train_loader,val_loader):
        
        model = Net_Individual(input_dimension=Input_Dimension,nodes_second = Nodes_Second,nodes_third = Nodes_Third,nodes_output = 1)   
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
       
        return model,val_losses

    def training_multi(self,Input_Dimension,Nodes_Second,Nodes_Third,Output_Dimension,Epoch,train_loader,val_loader,test_loader,representation_analysis,N):
        
        model = Net_Multi(input_dimension=Input_Dimension,nodes_second = Nodes_Second,nodes_third = Nodes_Third,nodes_output = Output_Dimension)    
        print(model)
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
                
                if Output_Dimension ==1 :
                    output = output.squeeze(1)
                    
        

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
                    
                    if Output_Dimension ==1 :
                        yhat = yhat.squeeze(1)
                        
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
                
           
            if representation_analysis:
                if np.mod(epoch,N)==0: 
                    Generate_Data = Representation_spatial_conceptual(model,test_loader)
                    Generate_Data.Isomap_plot_save(epoch)
       
        return model,val_losses
            

    

class Masks():
    
    def __init__(self,model):
        nodes_second = model.fc2[0].weight.shape[1]
        nodes_third = model.fc3[0].weight.shape[1]
        nodes_output = model.fc3[0].weight.shape[0]
        
        if nodes_output == 2:

            self.mask_conceptual_3 = torch.tensor( [[1 if i<nodes_third/2 else 0 for i in range(nodes_third)] if k<1
                                            else [0 for i in range(nodes_third)]
                                            for k in range(nodes_output)],
                                          dtype= torch.float64)


            self.mask_spatial_3 = torch.tensor( [[ 0 for i in range(nodes_third)] if k<1
                                            else [0 if i<nodes_third/2 else 1 for i in range(nodes_third)]
                                            for k in range(nodes_output)],
                                          dtype= torch.float64)
        if nodes_output == 1:

            self.mask_conceptual_3 = torch.tensor( [1 if i<nodes_third/2 else 0 for i in range(nodes_third)],
                                          dtype= torch.float64)


            self.mask_spatial_3 = torch.tensor( [0 if i<nodes_third/2 else 1 for i in range(nodes_third)],
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
        #print("pumpkin_seed",pumpkin_seed)        
        #print("fc3 weight grad",model.fc3[0].weight.grad)
        #print("fc2 weight grad",model.fc3[0].weight.grad)
                
                
                
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
 
        
