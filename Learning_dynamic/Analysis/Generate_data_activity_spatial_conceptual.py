import torch
import numpy as np
import sys
sys.path.append("/Users/charlottebeylier/Documents/PhD/Spatial and Conceptual Learning/github_code/Mono_Multi_TASK/Analysis")
from Hooks import Activation_Hook
sys.path.append("/Users/charlottebeylier/Documents/PhD/Atari1.0/Reinforcement-Learning_modif/cgames/02_space_invader/Experiments/Analysis")
from sklearn.manifold import Isomap



class Generate_Data():
    
    
    def __init__(self,model,test_loader):
       
        self.model = model
        self.test_loader = test_loader
        self.activation_hook = Activation_Hook()
        self.Names_hook = ["fc1","fc2","fc3"]
        
    
    def generate_preprocessed_data(self,multi): ##call preprocessing automatically
        
        self.activation_hook.registration(self.model)
        Liste_activation = [[] for i in range(len(self.Names_hook))] 
        Liste_seed = []
        episode = 0
        nodes_input = self.model.fc1[0].in_features

        for x_test,y_test in self.test_loader:
            with torch.no_grad():
                if x_test.shape[0]>1:
                    x_test = x_test.squeeze(0)
                yhat = torch.nn.Sigmoid()(self.model(x_test))
                Liste_seed.append(x_test[:,5].tolist())
                for h in range(len(self.Names_hook)):
                    b = self.activation_hook.activation[self.Names_hook[h]]
                    Liste_activation[h].append(b)
            
            episode += 1
        self.activation_hook.detach()
        activity_layer = self.preprocessing(Liste_activation)

        Liste_seed = torch.tensor(Liste_seed[:-1])  # same issue as in (****)
        Liste_seed = Liste_seed.flatten()
        
        return activity_layer,Liste_seed
    
    
    def preprocessing(self,Liste_activation):
        activity_layer = [[] for i in range(len(self.Names_hook))] 

        for layer in range(len(self.Names_hook)):   
            activity_layer[layer] = self.prepro(Liste_activation[layer])
        return activity_layer

    def prepro(self,Activation):
        
        activation = Activation
        Liste_activation = activation[0]
        
        for i in range(1,len(activation)-1):  # (****) by adding -1 we take care of the case where the batch size is not a multiple of the test dataset size (ie we have several batch size and then the last one is with whats left of the dataset). We dont have the problem of concatenating a tensor of size (batch_sizexnumber_of_nodes with Y-valuexnumber_of_nodes )
            Liste_activation = torch.cat((Liste_activation,activation[i]),0)  # if no unsqueeze then does not have the right shape (steps,nodes)
        return Liste_activation.cpu().detach().numpy()
    
    
    def generate_preprocessed_data_frozen(self,spatial): ##call preprocessing automatically
        
        self.activation_hook.registration(self.model)
        Liste_activation = [[] for i in range(len(self.Names_hook))] 
        episode = 0
        for x_test,y_test in self.test_loader:
            with torch.no_grad():
                pumpkin_seed = np.int(x_test[0][5])
                if not(spatial):
                    x_test = x_test[:,2:5]
                else:
                    x_test = x_test[:,0:2]
                yhat = torch.nn.Sigmoid()(self.model(x_test).squeeze(1))
                for h in range(len(self.Names_hook)):
                    b = torch.flatten(self.activation_hook.activation[self.Names_hook[h]])
                    Liste_activation[h].append(b)  
            
            episode += 1
        self.activation_hook.detach()
       
        activity_layer = self.preprocessing(Liste_activation)
        
        return activity_layer
    
   




        







































