import torch
import numpy as np
import torch.nn as nn


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
