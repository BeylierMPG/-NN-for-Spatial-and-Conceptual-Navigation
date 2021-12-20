import torch
import torch.nn as nn
import torch.nn.functional as F



class Net(torch.nn.Module):
    def __init__(self,input_dimension,multi_task):
        super(Net, self).__init__()

        if multi_task :
            nodes_second = 4 
            output_dimension = 2
        else:
            nodes_second = 2
            output_dimension = 1


        self.fc1 = nn.Linear(in_features = input_dimension, out_features = 2)
        self.fc2 = nn.Linear(in_features = 2, out_features = nodes_second)
        self.fc3 = nn.Linear(in_features = nodes_second,  out_features = output_dimension)

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        torch.nn.init.xavier_normal_(self.fc3.weight)


    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(self.fc2(x)) #tanh if does not work than not a smoothness issue
        x = self.fc3(x)
        return torch.sigmoid(x)



class Training():

    # The class initialisation function.
    def __init__(self,model,opti,loss):
        self.model = model
        self.optimizer = opti
        self.criterion = loss
    
    def train_network(self,data,target,pumpkin_seed,multi_task):

        self.mask1 = torch.tensor([[1,1,1,1,0],[1,1,1,1,0]])

        if pumpkin_seed == 1:
            self.mask3 = torch.tensor([[0,0,0,0],[0,0,1,1]])  

        if pumpkin_seed == 0:
            self.mask3 = torch.tensor([[1,1,0,0],[0,0,0,0]])

        with torch.no_grad():
            self.model.fc1.weight.mul_(self.mask1)

        Loss = []  
        #Compute the prediction
        output = self.model.forward(data)

        if len(output.shape) >len(target.shape):
            output = output.squeeze(1)
        # Calculate the loss for this transition.
        loss =  self.criterion(output,target)
        # Set all the gradients stored in the optimiser to zero.
        self.optimizer.zero_grad()

        loss.backward() 

        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        with torch.no_grad():
            self.model.fc1.weight.grad.mul_(self.mask1)
            if multi_task : 
                self.model.fc3.weight.grad.mul_(self.mask3) # still takes into account all the weights in the forward pass, will it just create a sort of bias?


        # Take one gradient step to update the Q-network. 
        self.optimizer.step()

        return loss



    