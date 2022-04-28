import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data.dataset import random_split


class Experiment:

    def __init__(self,width,batch_size,size_output_multi):
        self.width = width
        self.width_ones = 15
        self.batch_size = batch_size
        self.size_output_multi = size_output_multi

    def Generate_data(self):

        data = []
        data_space = []
        data_weather = []
        target = []
      
        for x in range(self.width):
            for y in range(self.width):
                data_space.append([x/self.width,y/self.width,np.random.randint(1,self.width)/self.width,np.random.randint(1,self.width)/self.width,np.random.randint(1,self.width)/self.width,1])

        for rain in range(self.width):
            for sun in range(self.width):
                for intensity in range(self.width):
                    data_weather.append([np.random.randint(1,self.width)/self.width,np.random.randint(1,self.width)/self.width,rain/self.width,sun/self.width,intensity/self.width,0])
        
        data_space = random.sample(data_space,len(data_space))
        data_weather = random.sample(data_weather,len(data_weather))
        data_weather = data_weather[0:len(data_space)] # to ensure the same number of data than for space
        

        target_space = [int(np.abs(a[1] - a[0]) <= self.width_ones/self.width) for a in data_space]
        
        target_weather = [int((np.abs(a[2] - a[3]) <= self.width_ones/self.width) and (np.abs(a[3] - a[4]) <= self.width_ones/self.width) and (np.abs(a[2] - a[4]) <= self.width_ones/self.width)) for a in data_weather]
        
        
        # MULTI TASK DATASET, TRAINING BLOCK OF SIZE BATCH SIZE Remplacer i par incidices
        for i in range(int(np.floor(len(data_space)/self.batch_size))):
            data.append([data_space[i*self.batch_size:(i+1)*self.batch_size]]) 
            data.append([data_weather[i*self.batch_size:(i+1)*self.batch_size]]) 

        
        data_space = torch.tensor(data_space,dtype = torch.float32)
        data_weather = torch.tensor(data_weather,dtype = torch.float32)
        data = torch.tensor(data,dtype = torch.float32).squeeze(1)
        data = torch.reshape(data,(data.shape[0]*data.shape[1],data.shape[2]))

        
        
        data_liste = data.tolist()
        
        if self.size_output_multi == 2:
            target = [[int(np.abs(a[1] - a[0]) <= self.width_ones/self.width),0] if a[5] == 0 else [0,int(np.abs(a[2] - a[3]) <= self.width_ones/self.width and np.abs(a[3] - a[4]) <= self.width_ones/self.width and np.abs(a[2] - a[4]) <= self.width_ones/self.width)] for a in data_liste] 
        
        if self.size_output_multi ==1 :
            target = [int(np.abs(a[1] - a[0]) <= self.width_ones/self.width) if a[5] == 0 else int(np.abs(a[2] - a[3]) <= self.width_ones/self.width and np.abs(a[3] - a[4]) <= self.width_ones/self.width and np.abs(a[2] - a[4]) <= self.width_ones/self.width) for a in data_liste] 


        
        target_space = torch.tensor(target_space,dtype = torch.float32)
        target_weather = torch.tensor(target_weather,dtype = torch.float32)
        target = torch.tensor(target,dtype = torch.float32)
        
        
        
        a = 0.7
        b = 0.9

        #dataset_train = TensorDataset(data[0:int(a*len(data))],target[0:int(a*len(data))])
        
        dataset_train = TensorDataset(data[0:int(a*len(data_space))],target[0:int(a*len(data_space))])
        dataset_space_train = TensorDataset(data_space[0:int(a*len(data_space))],target_space[0:int(a*len(data_space))])
        dataset_weather_train = TensorDataset(data_weather[0:int(a*len(data_weather))],target_weather[0:int(a*len(data_weather))])
        
        
        #dataset_val = TensorDataset(data[int(a*len(data))+1:int(b*len(data))],target[int(a*len(data))+1:int(b*len(data))])
        dataset_val = TensorDataset(data[int(a*len(data_space))+1:int(b*len(data_space))],target[int(a*len(data_space))+1:int(b*len(data_space))])
        dataset_space_val = TensorDataset(data_space[int(a*len(data_space))+1:int(b*len(data_space))],target_space[int(a*len(data_space))+1:int(b*len(data_space))])
        dataset_weather_val = TensorDataset(data_weather[int(a*len(data_weather))+1:int(b*len(data_weather))],target_weather[int(a*len(data_weather))+1:int(b*len(data_weather))])
        


       # dataset_test = TensorDataset(data[int(b*len(data))+1:len(data)],target[int(int(b*len(data)))+1:len(data)])
       
        dataset_test = TensorDataset(data[int(b*len(data_space))+1:len(data_space)],target[int(b*len(data_space))+1:len(data_space)])
        dataset_space_test = TensorDataset(data_space[int(b*len(data_space))+1:len(data_space)],target_space[int(b*len(data_space))+1:len(data_space)])
        dataset_weather_test = TensorDataset(data_weather[int(b*len(data_weather))+1:len(data_weather)],target_weather[int(b*len(data_weather))+1:len(data_weather)])
        
        print("Test size dataset space",data_space[int(b*len(data_space))+1:len(data_space)].shape)
        
        print("Test size dataset weather",data_weather[int(b*len(data_weather))+1:len(data_weather)].shape)
        
        #print("Test size dataset multi",data[int(b*len(data))+1:len(data)].shape)
        
        
        print("Train size dataset space",data_space[0:int(a*len(data_space))].shape)
        
        print("Train size dataset weather",data_weather[0:int(a*len(data_weather))].shape)
        
        #print("Train size dataset multi",data[0:int(a*len(data))].shape)
       # print("Train size dataset multi",data[0:int(a*len(data_weather))].shape)
        
        return dataset_train,dataset_space_train,dataset_weather_train,dataset_val,dataset_space_val,dataset_weather_val,dataset_test,dataset_space_test,dataset_weather_test

  


    



