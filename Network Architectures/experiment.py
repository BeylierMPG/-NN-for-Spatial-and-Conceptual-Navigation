import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data.dataset import random_split

class Experiment:

    def __init__(self,width,g_seed):

        # Set the width and height of the grid
        self.width = width
        self.Target_position()
        self.g_seed = g_seed
 
    def seed_worker():
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def Target_position(self):
        self.Target_position_grid = torch.eye(self.width,dtype=torch.int32)
        self.Target_position_grid[0,1]=1
        self.Target_position_grid[self.width-1,self.width-2]=1
        for i in range(1,self.width-1):
            self.Target_position_grid[i,i-1] = 1
            self.Target_position_grid[i,i+1] = 1



    def data_generator(self):
        x = np.random.randint(0,self.width)
        y = np.random.randint(0,self.width)
 
        sun = np.random.randn()
        rain = np.random.randn()
        seed = np.random.randint(0,2)
        data = [x,y,sun,rain,seed]  
        return data
    
    
    def Create_Dataset(self,SIZE,BATCH_SIZE):
        target = []
        data = [self.data_generator() for i in range(SIZE)]
        for a in data:
            if a[4] == 0:
                target.append(self.Target_position_grid[a[0],a[1]])
            else:
                if (a[2] == a[3] or a[2] == a[3]+1 or a[2] == a[3]-1):
                    target.append(1.0)
                else:
                    target.append(0.0)
                
        
        data, target = torch.tensor(data,dtype=torch.float32), torch.tensor(target,dtype=torch.float32)

        dataset = TensorDataset(data, target)
        train_dataset, val_dataset = random_split(dataset, [int(np.floor(0.8*SIZE)), SIZE-int(np.floor(0.8*SIZE))])
        train_loader = DataLoader(dataset = train_dataset, 
                                  batch_size = BATCH_SIZE, 
                                  worker_init_fn = self.seed_worker,
                                  generator = self.g_seed)
        val_loader = DataLoader(dataset = val_dataset, 
                                batch_size = BATCH_SIZE, 
                                worker_init_fn = self.seed_worker,
                                generator = self.g_seed)

        return train_loader,val_loader





