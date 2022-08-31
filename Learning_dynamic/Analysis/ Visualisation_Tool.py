
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
import cv2
import glob
import pandas as pd
from scipy.spatial import distance_matrix


class Visu_Tool():
    def __init__(self,network_type):
        
        
        self.Names_hook = ["fc1","fc2","fc3"]
        self.general_path = "/Users/charlottebeylier/Documents/PhD/Spatial and Conceptual Learning/github_code/Learning_dynamic/Result_learning_representation"
        self.creation_files(network_type)

    def creation_files(self,network_type):
        try:
            os.mkdir(os.path.join(self.general_path,network_type))
        except OSError as error:
            pass

        # for layer in range(len(self.Names_hook)):
        #     try:
        #         os.mkdir(os.path.join(self.general_path,network_type,self.Names_hook[layer]))      
        #     except OSError as error:
        #         pass
        
        return os.path.join(self.general_path,network_type)

    def movie(self,path_images):
        frameSize = (500, 500)
        out = cv2.VideoWriter('output_video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 60, frameSize)
        for filename in glob.glob(path_images+'*.jpg'):
            img = cv2.imread(filename)
            out.write(img)
        out.release()





    




