import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data.dataset import random_split
from sklearn.model_selection import KFold


class Experiment:

    def __init__(self,width,batch_size):
        self.width = width
        self.batch_size = batch_size
        self.dataset,self.dataset_space,self.dataset_weather= self.Generate_data()

    def Generate_data(self):

        data = []
        data_space = []
        data_weather = []
      
        for x in range(self.width):
            for y in range(self.width):
                data_space.append([x/self.width,y/self.width,np.random.rand(1)[0],np.random.rand(1)[0],1])

        for rain in range(self.width):
            for sun in range(self.width):
                data_weather.append([np.random.rand(1)[0],np.random.rand(1)[0],rain/self.width,sun/self.width,0])
        
        data_space = random.sample(data_space,len(data_space))
        data_weather = random.sample(data_weather,len(data_weather))

        # MULTI TASK DATASET, TRAINING BLOCK OF SIZE BATCH SIZE 
        for i in range(int(np.floor(len(data_space)/self.batch_size))):
            data.append(data_space[i*self.batch_size:(i+1)*self.batch_size]) 
            data.append(data_weather[i*self.batch_size:(i+1)*self.batch_size])    
        
        
        
        data_space = torch.tensor(data_space,dtype=torch.float32)
        data_weather = torch.tensor(data_weather,dtype=torch.float32)
        data = torch.tensor(data,dtype=torch.float32)
        data = torch.reshape(data,[data.shape[0]*data.shape[1],data.shape[2]])

        # 
        target = [[int(np.abs(a[1] - a[0]) <= 34/self.width),0] if a[4] == 1 else [0,int(np.abs(a[2] - a[3]) <= 34/self.width)] for a in data]
        target_space = [int(np.abs(a[1] - a[0]) <= 34/self.width) if a[4] == 1 else int(np.abs(a[2] - a[3]) <= 34/self.width) for a in data_space]
        target_weather = [int(np.abs(a[1] - a[0]) <= 34/self.width) if a[4] == 1 else int(np.abs(a[2] - a[3]) <= 34/self.width) for a in data_weather]

        target = torch.tensor(target,dtype=torch.float32)
        target_space = torch.tensor(target_space,dtype=torch.float32)
        target_weather = torch.tensor(target_weather,dtype=torch.float32)


        dataset = TensorDataset(data,target)
        dataset_space = TensorDataset(data_space,target_space)
        dataset_weather = TensorDataset(data_weather,target_weather)

        return dataset,dataset_space,dataset_weather

    def Create_dataset(self,multi_task,space,weather):

        if multi_task:
            final_dataset = self.dataset
        elif space:
            final_dataset = self.dataset_space
        elif weather:
            final_dataset = self.dataset_weather

   
        train_loader = torch.utils.data.DataLoader(final_dataset,batch_size=self.batch_size, sampler=None)

        test_loader = torch.utils.data.DataLoader(final_dataset,batch_size=self.batch_size, sampler=None)

        
        return train_loader, test_loader




