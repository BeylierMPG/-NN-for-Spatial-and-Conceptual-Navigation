
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
        
        pass

    
    def dissimilarity_matrix(self,X,liste_seed):
        index = sorted(range(len(liste_seed)), key=lambda k: liste_seed[k])
        sorted_X = [X[i] for i in index]
        df = pd.DataFrame(sorted_X)
        Distance_matrix = pd.DataFrame(distance_matrix(df.values,df.values))
        Distance_matrix = Distance_matrix.round(decimals=1,out=None)

        return Distance_matrix