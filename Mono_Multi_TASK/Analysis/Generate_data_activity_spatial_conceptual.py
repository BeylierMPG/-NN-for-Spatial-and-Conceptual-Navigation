import torch
import numpy as np
import sys
sys.path.append("/Users/charlottebeylier/Documents/PhD/Spatial and Conceptual Learning/github_code/Mono_Multi_TASK/Analysis")
from Hooks import Activation_Hook
sys.path.append("/Users/charlottebeylier/Documents/PhD/Atari1.0/Reinforcement-Learning_modif/cgames/02_space_invader/Experiments/Analysis")
from Manifold_Analysis import Manifold_analysis




class Generate_Data():
    
    
    def __init__(self,model,test_loader):
       
        self.model = model
        self.test_loader = test_loader
        self.activation_hook = Activation_Hook()
        self.Names_hook = ["fc1","fc2","fc3"]
        
    
    def preprocessing(self,Liste_activation):
        analysis = Manifold_analysis(length_trial = len(Liste_activation))
        Prepro_length = False
        activity_layer = [[] for i in range(len(self.Names_hook))] 

        for layer in range(len(self.Names_hook)):
            activity_layer[layer] = analysis.prepro(Liste_activation[layer],Prepro_length)
            
        return activity_layer
    
    def generate_preprocessed_data(self,multi): ##call preprocessing automatically
        
        self.activation_hook.registration(self.model)
        Liste_activation = [[] for i in range(len(self.Names_hook))] 
        episode = 0
        nodes_input = self.model.fc1[0].in_features
        print("ggggg")
        print("nodes input",nodes_input)
        for x_test,y_test in self.test_loader:
            with torch.no_grad():
                pumpkin_seed = np.int(x_test[0][5])
                if nodes_input == 2:
                    x_test = x_test[:,0:2]
               
                elif nodes_input == 3:
                    x_test = x_test[:,2:5]
                else:
                    x_test = x_test.squeeze(0)
                yhat = torch.nn.Sigmoid()(self.model(x_test))
                for h in range(len(self.Names_hook)):
                    b = torch.flatten(self.activation_hook.activation[self.Names_hook[h]])
                    Liste_activation[h].append(b)  
            
            episode += 1
        self.activation_hook.detach()
       
        activity_layer = self.preprocessing(Liste_activation)
        
        return activity_layer
    
    
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
    
   




        







































