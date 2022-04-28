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


class Training():
    
    def __init__(self):
        self.Liste_cka =[[] for i in range(3)] # 3 corresponds to the number of hidden layers
        self.Liste_cka_init =[[] for i in range(3)]

    
    def training_individual(self,Input_Dimension,Nodes_Second,Nodes_Third,Epoch,train_loader,val_loader,test_loader,do_analysis,type,network_type,frequence):
        
        model = Net_Individual(input_dimension=Input_Dimension,nodes_second = Nodes_Second,nodes_third = Nodes_Third,nodes_output = 1)   
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        criterion = nn.BCEWithLogitsLoss()


        losses = []
        val_losses = []

        for epoch in range(Epoch):

            loss = 0
            val_loss = 0 

            for x_batch, y_batch in train_loader:

                x_batch = x_batch[:,0:5]

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
                    x_val = x_val[:,0:5]
                    yhat = model(x_val).squeeze(1)
                    val_loss += criterion(yhat,y_val)
                val_losses.append(val_loss.item()/len(val_loader))


            if np.mod(epoch,10)==0: 
                clear_output()
                print("Epoch {}, val_loss {}".format(epoch, val_loss.item()/len(val_loader))) 
                fig = plt.figure()
                ax = fig.add_subplot(111)
                plt.plot(np.arange(len(val_losses)), val_losses)
                plt.ylabel('Loss')
                plt.xlabel('Epoch #')
                plt.show()


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

    def training_multi(self,Input_Dimension,Nodes_Second,Nodes_Third,Output_Dimension,Epoch,train_loader,val_loader,test_loader,do_analysis,type,frequence):
        
        model = Net_Multi(input_dimension=Input_Dimension,nodes_second = Nodes_Second,nodes_third = Nodes_Third,nodes_output = Output_Dimension)    
        print(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        criterion = nn.BCEWithLogitsLoss()
        mask = Masks(model)

        
        losses = []
        val_losses = []
        
        for epoch in range(Epoch):

            loss = 0
            val_loss = 0 

            for x_batch, y_batch in train_loader:


                pumpkin_seed= x_batch[0,5]
                x_batch = x_batch.squeeze(0)

                optimizer.zero_grad()
                
                output = model(x_batch)
                
                if Output_Dimension ==1 :
                    output = output.squeeze(1)
                    
        

                output_loss = criterion(output,y_batch.squeeze(0))
                output_loss.backward()

                mask.apply_mask(model,pumpkin_seed)                    
                optimizer.step()

                loss += output_loss.detach().numpy()
            losses.append(loss/len(train_loader))

            with torch.no_grad():
                for x_val, y_val in val_loader:
                    model.eval() 
                    x_val = x_val.squeeze(0)
                    yhat = model(x_val)
                    
                    if Output_Dimension ==1 :
                        yhat = yhat.squeeze(1)
                        
                    val_loss += criterion(yhat,y_val.squeeze(0))
                val_losses.append(val_loss.item()/len(val_loader))


            if np.mod(epoch,10)==0: 
                clear_output()
                print("Epoch {}, val_loss {}".format(epoch, val_loss.item()/len(val_loader))) 
                fig = plt.figure()
                ax = fig.add_subplot(111)
                plt.plot(np.arange(len(val_losses)), val_losses)
                plt.ylabel('Loss')
                plt.xlabel('Epoch #')
                plt.show()
                
           
        
            
            if do_analysis:
          
                if np.mod(epoch,frequence)==0: 
                    generate_data = Generate_Data(model,test_loader)
                    activity_layer = generate_data.generate_preprocessed_data(multi = True)

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
                        representation_learning = Representation_Learning(frequence,"Multi")
                        representation_learning.Isomap_plot_save(activity_layer,epoch,"Multi")
                        


       
        return model,val_losses
            
