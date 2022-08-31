import torch
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output
import torch.nn as nn
import sys
sys.path.append("/Users/charlottebeylier/Documents/PhD/Spatial and Conceptual Learning/github_code/Learning_dynamic/Analysis")
from Generate_data_activity_spatial_conceptual import Generate_Data
from Representation_learning import Representation_Learning

sys.path.append("/Users/charlottebeylier/Documents/PhD/Spatial and Conceptual Learning/github_code/Learning_dynamic/Models")
from Networks_v2 import Net_Multi
from Mask import Masks
from sklearn.manifold import Isomap


class Training():
    
    def __init__(self):
        self.Liste_cka =[[] for i in range(3)] # 3 corresponds to the number of hidden layers
        self.Liste_cka_init =[[] for i in range(3)]
        self.Names_hook = ["fc1","fc2","fc3"]

    def training_multi(self,Input_Dimension,Nodes_Second,Nodes_Third,Output_Dimension,Epoch,train_loader,val_loader,test_loader,do_analysis,type = 1,frequence=10):
        
        model = Net_Multi(input_dimension=Input_Dimension,nodes_second = Nodes_Second,nodes_third = Nodes_Third,nodes_output = Output_Dimension)    
        #print(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        criterion = nn.BCEWithLogitsLoss()
        mask = Masks(model)

        # print("Weights init fc3",model.fc3[0].weight)
        # print("Weights init fc2",model.fc2[0].weight)

        
        losses = []
        val_losses = []
        
        for epoch in range(Epoch):

            loss = 0
            val_loss = 0 

            for x_batch, y_batch in train_loader:
                pumpkin_seed= x_batch[0,5]
                x_batch = x_batch.squeeze(0)
                #print("X:batch")
                #print(x_batch[0:10])
                optimizer.zero_grad()
                output = model(x_batch)
                
                if Output_Dimension ==1 :
                    output = output.squeeze(1)
                    
                output_loss = criterion(output,y_batch.squeeze(0))
                output_loss.backward()

                mask.apply_mask(model,pumpkin_seed)
                # print("--------")
                # print("pumpkin_seed",pumpkin_seed)
                # print("Weights mask fc3",model.fc3[0].weight.grad)
                # print("Weights mask fc2",model.fc2[0].weight.grad)   
                # print("--------")             
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
                #clear_output()
                print("Epoch {}, val_loss {}".format(epoch, val_loss.item()/len(val_loader))) 

                generate_data = Generate_Data(model,test_loader)
                activity_layer,Liste_seed = generate_data.generate_preprocessed_data(multi = False)
                color_map = ["green"if i==0 else "blue" for i in Liste_seed ]
                rep_learning = Representation_Learning()

                fig = plt.figure(figsize=(12,10))
                ax = fig.add_subplot(3,1,3)
                ax.plot(np.arange(len(val_losses)), val_losses)
                plt.ylabel('Loss')
                plt.xlabel('Epoch #')
              
                if do_analysis:
                    for layer in range(len(self.Names_hook)):

                        ax = fig.add_subplot(3,3,layer+1,projection='3d')
                        embedding = Isomap(n_neighbors=20,n_components=3)
                        X = embedding.fit_transform(activity_layer[layer])
                        ax.scatter3D(X[:, 0], X[:, 1], X[:, 2],c = color_map)
                        plt.title("Layer:"+str(self.Names_hook[layer]))


                        DSM = rep_learning.dissimilarity_matrix(X,Liste_seed)
                        ax = fig.add_subplot(3,3,layer+1+3)
                        cax = ax.matshow(DSM, interpolation='nearest')
                        ax.grid(True)
                    fig.colorbar(cax)
                
                
                plt.show()
                    
                
        


       
        return model,val_losses
            

            
