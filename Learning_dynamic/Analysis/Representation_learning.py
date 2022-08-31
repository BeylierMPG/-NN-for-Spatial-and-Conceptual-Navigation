
import torch
import numpy as np
import os
import sys
sys.path.append("/Users/charlottebeylier/Documents/PhD/Spatial and Conceptual Learning/github_code/Mono_Multi_TASK/Analysis")
from Hooks import Activation_Hook
sys.path.append("/Users/charlottebeylier/Documents/PhD/Atari1.0/Reinforcement-Learning_modif/cgames/02_space_invader/Analysis")
from sklearn.manifold import Isomap
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance

import pandas as pd
from scipy.spatial import distance_matrix


class Representation_Learning():
    def __init__(self,frequence_image_saving= 10,network_type= "Multi"):
        
        self.N = frequence_image_saving
        self.Names_hook = ["fc1","fc2","fc3"]
        self.general_path = "/Users/charlottebeylier/Documents/PhD/Spatial and Conceptual Learning/github_code/Mono_Multi_TASK/Result_learning_representation"
        self.creation_files(network_type)

    def creation_files(self,network_type):
        try:
            os.mkdir(os.path.join(self.general_path,network_type))
        except OSError as error:
            pass

        for layer in range(len(self.Names_hook)):
            try:
                os.mkdir(os.path.join(self.general_path,network_type,self.Names_hook[layer]))      
            except OSError as error:
                pass

    def Isomap_plot_save(self,activity_layer,epoch,network_type): ##call generate data and preprocessing automatically
            
            for layer in range(len(self.Names_hook)):
                fig = plt.figure(figsize=(10,10))
                ax = fig.add_subplot(1,1,1,projection='3d')
                embedding = Isomap(n_neighbors=15,n_components=3)
                X = embedding.fit_transform(activity_layer[layer])
                ax.scatter3D(X[:, 0], X[:, 1], X[:, 2])
                path = os.path.join(self.general_path,network_type,self.Names_hook[layer],"step_"+str(epoch)+".jpg")
                print(path)
                plt.savefig(path)



    def dissimilarity_matrix(self,X,liste_seed):
        index = sorted(range(len(liste_seed)), key=lambda k: liste_seed[k])
        sorted_X = [X[i] for i in index]
        df = pd.DataFrame(sorted_X)
        Distance_matrix = pd.DataFrame(distance_matrix(df.values,df.values))
        Distance_matrix = Distance_matrix.round(decimals=1,out=None)

        return Distance_matrix




