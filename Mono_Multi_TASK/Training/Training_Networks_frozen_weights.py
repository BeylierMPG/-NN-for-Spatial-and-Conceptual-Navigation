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

    
    def training_individual(self,model,optimizer,criterion,Epoch,train_loader,val_loader,test_loader,spatial_task,option_interleaved,do_analysis,type,network_type,frequence):
        
        


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

   