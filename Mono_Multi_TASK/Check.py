import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data.dataset import random_split

import torch.nn as nn
import time
from matplotlib import pyplot as plt






class Checkup():
    def __init__(self):
        
        pass
    
    
    def Display_Score(self,model,test_loader,epsilon,example,multi):
        length = 0
        score = 0
        for x_test,y_test in test_loader:
             with torch.no_grad():
                pumpkin_seed = np.int(x_test[0][5])
                x_test = x_test[:,0:5]
                yhat = torch.nn.Sigmoid()(model(x_test).squeeze(1))
                length += len(y_test)
                    
                for i in range(len(yhat)):
                    #take the prediction for the output node corresponding to either the conceptual or the spatial task
                    if multi:
                        if np.abs(yhat[i][pumpkin_seed] - y_test[i][pumpkin_seed] )<epsilon:
                            score+=1
                    else:
                        if np.abs(yhat[i]- y_test[i] )<epsilon:
                            score+=1


        if example:
            print("Data",x_test)
            print("Target",y_test)
            print("Prediction",yhat)
        print("Score final : ", (score/length)*100)
        
    
