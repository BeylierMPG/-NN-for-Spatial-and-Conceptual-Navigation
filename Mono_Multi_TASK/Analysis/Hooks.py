
class Activation_Hook():
    
    def __init__(self):
        self.activation = {}
        
        
    def getActivation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook
            
    
    def registration(self,model):
        
        self.h1 = model.fc1.register_forward_hook(self.getActivation('fc1'))
        self.h2 = model.fc2.register_forward_hook(self.getActivation('fc2'))
        self.h3 = model.fc3.register_forward_hook(self.getActivation('fc3'))
        
        

    def detach(self):
        self.h1.remove()
        self.h2.remove()
        self.h3.remove()