import torch
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output
import torch.nn as nn
import sys
sys.path.append("/Users/charlottebeylier/Documents/PhD/Spatial and Conceptual Learning/github_code/Mono_Multi_TASK/Analysis")
from Generate_data_activity_spatial_conceptual import Generate_Data
from Representation_learning import Representation_Learning
from Similarity_of_Neural_Network_Representations_Revisited import feature_space_linear_cka

sys.path.append("/Users/charlottebeylier/Documents/PhD/Spatial and Conceptual Learning/github_code/Mono_Multi_TASK/Models")
from Networks_v2 import Net_Multi,Net_Individual
from Mask import Masks


class Training_frozen_weights():
    
    def __init__(self):
        self.Liste_cka =[[] for i in range(3)] # 3 corresponds to the number of hidden layers
        self.Liste_cka_init =[[] for i in range(3)]
        
        self.Liste_cka_weather =[[] for i in range(3)] # 3 corresponds to the number of hidden layers
        self.Liste_cka_init_weather =[[] for i in range(3)]
        
        self.Liste_cka_space =[[] for i in range(3)] # 3 corresponds to the number of hidden layers
        self.Liste_cka_init_space =[[] for i in range(3)]
        

    
    def training_individual(self,model,optimizer,criterion,Epoch,train_loader,val_loader,test_loader,spatial_task,option_interleaved,epoch_interleaved,do_analysis,type,network_type,frequence):
        
        


        losses = []
        val_losses = []

        for epoch in range(Epoch):

            loss = 0
            val_loss = 0 

            for x_batch, y_batch in train_loader:
                if spatial_task:
                    x_batch = x_batch[:,0:2]
                else:
                    x_batch = x_batch[:,2:5]

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
                    if spatial_task:
                        x_val = x_val[:,0:2]
                    else:
                        x_val = x_val[:,2:5]
                    yhat = model(x_val).squeeze(1)
                    val_loss += criterion(yhat,y_val)
                val_losses.append(val_loss.item()/len(val_loader))


    

            if do_analysis:
                
                if np.mod(epoch,frequence)==0: 
                    generate_data = Generate_Data(model,test_loader)
                    activity_layer = generate_data.generate_preprocessed_data(multi = False)

                    if type == 0:
                        if epoch == 0:
                            self.activity_layer_init = activity_layer
                            self.activity_layer_t_prime = activity_layer
                        else:
                            self.activity_layer_t = self.activity_layer_t_prime
                            self.activity_layer_t_prime = activity_layer

                            for layer in range(3):

                                cka = feature_space_linear_cka(self.activity_layer_t[layer], self.activity_layer_t_prime[layer], debiased=False)
                                cka_init =feature_space_linear_cka(self.activity_layer_init[layer], self.activity_layer_t_prime[layer], debiased=False)    
                                
                                for g in range(frequence):
                                    self.Liste_cka[layer].append(cka)
                                    self.Liste_cka_init[layer].append(cka_init)

                    if type == 1:
                        representation_learning = Representation_Learning(frequence, network_type)
                        representation_learning.Isomap_plot_save(activity_layer,epoch,network_type)

                    

       
        return model,val_losses
    

    
    
    
    
    
    
    
    def training_individual_interleaved(self,Epoch,Nodes_Second,Nodes_Third,train_loader_weather,val_loader_weather,test_loader_weather,train_loader_space,val_loader_space,test_loader_space,do_analysis,type,network_type,frequence):
        
        
        Input_Dimension = 3
        model_weather = Net_Individual(input_dimension=3,nodes_second = Nodes_Second,nodes_third = Nodes_Third,nodes_output = 1)  
        optimizer_weather = torch.optim.Adam(model_weather.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        Input_Dimension = 2
        model_space = Net_Individual(Input_Dimension ,Nodes_Second,Nodes_Third,1)    
        optimizer_space = torch.optim.Adam(model_space.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

        criterion = nn.BCEWithLogitsLoss()
        


        losses_weather = []
        val_losses_weather = []
        
        losses_space = []
        val_losses_space = []

        for epoch in range(Epoch):

            loss = 0
            val_loss = 0 

            for x_batch, y_batch in train_loader_weather:
                x_batch = x_batch[:,2:5]

                optimizer_weather.zero_grad()
                output = model_weather(x_batch).squeeze(1)

                output_loss = criterion(output,y_batch)
                output_loss.backward()
                optimizer_weather.step()

                loss += output_loss.detach().numpy()
            losses_weather.append(loss/len(train_loader_weather))

            with torch.no_grad():
                for x_val, y_val in val_loader_weather:
                    model_weather.eval() 
                    x_val = x_val[:,2:5]
                    yhat = model_weather(x_val).squeeze(1)
                    val_loss += criterion(yhat,y_val)
                val_losses_weather.append(val_loss.item()/len(val_loader_weather))
                
            
            
            
            model_space.fc2[0].weight = model_weather.fc2[0].weight
            model_space.fc3[0].weight = model_weather.fc3[0].weight

            for param in model_space.fc2.parameters():
                param.requires_grad = False

            for param in model_space.fc3.parameters():
                param.requires_grad = False
            
            
            loss = 0
            val = 0 

            for x_batch, y_batch in train_loader_space:
                x_batch = x_batch[:,0:2]
                

                optimizer_space.zero_grad()
                output = model_space(x_batch).squeeze(1)

                output_loss = criterion(output,y_batch)
                output_loss.backward()
                optimizer_space.step()

                loss += output_loss.detach().numpy()
            losses_space.append(loss/len(train_loader_space))

            with torch.no_grad():
                for x_val, y_val in val_loader_space:
                    model_space.eval() 
                    x_val = x_val[:,0:2]
                   
                    yhat = model_space(x_val).squeeze(1)
                    val_loss += criterion(yhat,y_val)
                val_losses_space.append(val_loss.item()/len(val_loader_space))


    

            if do_analysis:
                
                if np.mod(epoch,frequence)==0: 
                    generate_data_weather = Generate_Data(model_weather,test_loader_weather)
                    activity_layer_weather = generate_data_weather.generate_preprocessed_data(multi = False)
                    
                    generate_data_space = Generate_Data(model_space,test_loader_space)
                    activity_layer_space = generate_data_space.generate_preprocessed_data(multi = False)

                    if type == 0:
                        if epoch == 0:
                            self.activity_layer_init_weather = activity_layer_weather
                            self.activity_layer_t_prime_weather = activity_layer_weather
                            
                            self.activity_layer_init_space = activity_layer_space
                            self.activity_layer_t_prime_space = activity_layer_space
                        else:
                            self.activity_layer_t_weather = self.activity_layer_t_prime_weather
                            self.activity_layer_t_prime_weather = activity_layer_weather
                            
                            self.activity_layer_t_space = self.activity_layer_t_prime_space
                            self.activity_layer_t_prime_space = activity_layer_space

                            for layer in range(3):

                                cka_weather = feature_space_linear_cka(self.activity_layer_t_weather[layer], self.activity_layer_t_prime_weather[layer], debiased=False)
                                cka_init_weather =feature_space_linear_cka(self.activity_layer_init_weather[layer], self.activity_layer_t_prime_weather[layer], debiased=False)  
                                
                                cka_space = feature_space_linear_cka(self.activity_layer_t_space[layer], self.activity_layer_t_prime_space[layer], debiased=False)
                                cka_init_space =feature_space_linear_cka(self.activity_layer_init_space[layer], self.activity_layer_t_prime_space[layer], debiased=False) 
                                
                                for g in range(frequence):
                                    self.Liste_cka_weather[layer].append(cka_weather)
                                    self.Liste_cka_init_weather[layer].append(cka_init_weather)
                                    
                                    self.Liste_cka_space[layer].append(cka_space)
                                    self.Liste_cka_init_space[layer].append(cka_init_space)

                    if type == 1:
                        representation_learning = Representation_Learning(frequence, network_type)
                        representation_learning.Isomap_plot_save(activity_layer,epoch,network_type)

                    

       
        return model_space,val_losses_space,model_weather,val_losses_weather
    
   