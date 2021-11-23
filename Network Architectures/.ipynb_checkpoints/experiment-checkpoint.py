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
        self.Target_position_grid = np.eye(self.width)
        self.Target_position_grid[0,1]=1
        self.Target_position_grid[self.width-1,self.width-2]=1
        for i in range(1,self.width-1):
            self.Target_position_grid[i,i-1] = 1
            self.Target_position_grid[i,i+1] = 1



    def data_generator_training(self,type_of_seed):
        x = np.random.randint(0,self.width)
        y = np.random.randint(0,self.width)
 
        sun = np.random.randint(0,self.width)
        rain = np.random.randint(0,self.width)
        pumpkin_seed = type_of_seed
        data = [x,y,sun,rain,pumpkin_seed]  
        return data
    
    def data_generator_evaluation(self):
        x = np.random.randint(self.width,self.width + 4)
        y = np.random.randint(self.width,self.width + 4)
 
        sun = np.random.randint(self.width,self.width + 4)
        rain = np.random.randint(self.width,self.width + 4)
        pumpkin_seed = np.random.randint(0,1)
        data = [x,y,sun,rain,pumpkin_seed]  
        return data
    
    
    
    def Create_Dataset(self,NUMBER_OF_BLOCKS,NUMBER_OF_TRIALS):
        target = []
        #data = [self.data_generator(type_of_seed = np.mod(i,2)) for i in range(SIZE)]
        data = []
        data_evaluation = []
        target_evaluation = []
        for k in range(NUMBER_OF_BLOCKS):
            for i in range(NUMBER_OF_TRIALS):
                data.append(self.data_generator_training(type_of_seed = np.mod(k,2)))
                data_evaluation.append(self.data_generator_evaluation())
        
        print("data_evaluation",data_evaluation)
        for a in data:
            if a[4] == 1:
                target.append([self.Target_position_grid[a[0],a[1]],0.0])
            else:
                if (a[2] == a[3] or a[2] == a[3]+1 or a[2] == a[3]-1):
                    target.append([0.0,1.0])
                else:
                    target.append([0.0,0.0])
        
                                       
        for a in data_evaluation :
            if a[4] == 1:
                target_evaluation.append([self.Target_position_grid[a[0],a[1]],0.0])
            else:
                if (a[2] == a[3] or a[2] == a[3]+1 or a[2] == a[3]-1):
                    target_evaluation.append([0.0,1.0])
                else:
                    target_evaluation.append([0.0,0.0])
                                       
        print("target_evaluation",target_evaluation)
        data, target = torch.tensor(data,dtype=torch.float32), torch.tensor(target,dtype=torch.float32)
        data_evaluation, target_evaluation = torch.tensor(data_evaluation,dtype=torch.float32), torch.tensor(target_evaluation,dtype=torch.float32)


        train_dataset = TensorDataset(data, target)
        val_dataset = TensorDataset(data_evaluation, target_evaluation)
        #train_dataset, val_dataset = random_split(dataset, [int(np.floor(0.8*SIZE)), SIZE-int(np.floor(0.8*SIZE))])
        train_loader = DataLoader(dataset = train_dataset, 
                                  batch_size = NUMBER_OF_TRIALS, 
                                  worker_init_fn = self.seed_worker,
                                  generator = self.g_seed)
        val_loader = DataLoader(dataset = val_dataset, 
                                batch_size = NUMBER_OF_TRIALS, 
                                worker_init_fn = self.seed_worker,
                                generator = self.g_seed)
        

        return train_loader,val_loader





