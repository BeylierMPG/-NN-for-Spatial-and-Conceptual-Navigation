import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import cv2
import random
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data.dataset import random_split




def print_params(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)





def layers_connexion(model,input_size):
    Layers = 1 #input layer
    Connexions = []
    Nodes = [input_size]
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "bias" in name:
                Layers += 1
                Nodes.append(len(param.detach().numpy()))
            else:
                a = param.detach().numpy()
                
            #Replace weight value by binary number
                a = np.select( [a != 0.0, a== 0.0], [1,0],default = a)
                a = np.array(a)
                Connexions.append(a)
                #print("Connexions",Connexions)
    return(Layers, Connexions, Nodes)




class Network_Visualization:
    def __init__(self,display,magnification,Model,input_size):
        self.layers,self.connexions,self.nodes = layers_connexion(Model,input_size)
        print(self.connexions)
        #self.connexions = [np.array([[1., 1., 1., 1., 1.],[1., 1., 1., 1., 1.]]),np.array([[2., 2.],[2., 2.],[3., 3.],[3., 3.]]), np.array([[2., 2., 3., 3.]])]
        # Set whether the environment should be displayed after every step
        self.display = display
        # Set the magnification factor of the display
        self.magnification = magnification
        # Create an image which will be used to display the environment
        self.image = np.zeros([int(self.magnification), int(self.magnification), 3], dtype=np.uint8)
        # Set the width and height of the environment
        self.width = 5
        self.height = 5
        self.rayon = 20
            
        # Init y coordinates of each layers
        self.b = np.linspace(4* self.rayon,
                        self.magnification-4*self.rayon,
                        self.layers)
        # Init coordinates of each nodes, i//self.layers allow to have the same y coordinates for every nodes of the same layer
        self.nodes_coordinates()

    def nodes_coordinates(self):
        self.Nodes_coordinates = []
        N_max = np.argmax(self.nodes)  
        a = np.linspace(4* self.rayon, self.magnification-4*self.rayon, self.nodes[N_max])
        inter_space = a[1] - a[0]

        for i in range(self.layers):
            width_bound = self.nodes[i]*2+self.rayon + (self.nodes[i] - 1)*inter_space
            lower_bound = int(np.floor((self.magnification- width_bound)/2))
            upper_bound = int(np.floor((self.magnification + width_bound)/2))
           # print("lower_bound",lower_bound)
           # print(max([4* self.rayon,lower_bound]))
            a = np.linspace(np.max([4* self.rayon,lower_bound]),
                            np.min([self.magnification-4*self.rayon,upper_bound]),
                            self.nodes[i])
            coordinates = [[int(a[j]),int(self.b[i])] for j in range(len(a))]
            self.Nodes_coordinates.append(coordinates)    
    
    def draw(self):
        # Create the background image
        window_top_left = (0, 0)
        window_bottom_right = (self.magnification , self.magnification )
        cv2.rectangle(self.image, window_top_left, window_bottom_right, (255, 255, 255), thickness=cv2.FILLED)
        #Draw the Nodes
        
        for i in range(self.layers):
            for j in range(len(self.Nodes_coordinates[i])):
                cv2.circle(self.image,center= self.Nodes_coordinates[i][j], 
                           radius = self.rayon, 
                           color =(0,0,0), 
                           thickness = 2 )
        #Draw the connexions
        for i in reversed(range(1, self.layers )):
            g = self.connexions[i-1].shape
            for j in range(g[0]):
                for k in range(g[1]):
                    if self.connexions[i-1][j][k] == 1:
                        COLOR = (0,0,0)
                    if self.connexions[i-1][j][k] == 2:
                        COLOR = (200,0,0)
                    if self.connexions[i-1][j][k] == 3:
                        COLOR = (0,0,200)    
                    cv2.line(self.image, 
                            pt1 = np.array(self.Nodes_coordinates[i-1][k]) + [0,self.rayon], 
                            pt2 = np.array(self.Nodes_coordinates[i][j]) - [0,self.rayon],
                            color =  COLOR , 
                            thickness = 2)

        cv2.imshow("Neural Network", self.image)
        # This line is necessary to give time for the image to be rendered on the screen
        cv2.waitKey(1)