import torch
import numpy as np
import torch.nn as nn



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
    
    


    