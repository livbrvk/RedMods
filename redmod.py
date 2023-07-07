import rnn
import torch
import numpy as np

import torch.nn as nn
import copy

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    

class PCE(nn.Module):
    #we get r from some of the values of the projection matrix
    def __init__(self, r, p=2):
        super(PCE,self).__init__()
        self.r = r
        
        if device is None:
            self.device = 'cpu'
        else:
            self.device = device
        
        
    def forward(self,x):
        return rnn.get_PCE_container(x.detach(),r=self.r)


class RedMod(nn.Module):
    
    def __init__(self, redmod_dict, pce_fnc = PCE):
        
        super(RedMod, self).__init__()
        
        pre = redmod_dict['pre']
        proj_mat = redmod_dict['proj_mat']
        inout_map = redmod_dict['inout_map']
        
        p=2
        n_classes=10
        r = proj_mat.size()[1]
        
        #pre-model is an FNN
        self.pre = pre
        
        #projection layer is a linear layer with weights equivalent to the projection matrix
        self.proj = nn.Linear(proj_mat.size()[0], proj_mat.size()[1], bias = False)
        self.proj.weight.data = copy.deepcopy(proj_mat).T #not sure if i need .t() here
        
        #Checking whether this may be PCE. We pass in the coefficients for PCE
        if isinstance(inout_map, np.ndarray):
            self.inout_basis = pce_fnc(r=r)
            self.inout_lay = nn.Linear(int(np.math.factorial(r+p)/(np.math.factorial(p)*np.math.factorial(r))), n_classes,
                                           bias=False)
            self.inout_lay.weight.data = copy.deepcopy(torch.from_numpy(inout_map)).t()
            self.inout_map = nn.Sequential(self.inout_basis, self.inout_lay)
            
        #If not, it is definitely FNN
        else:
            self.inout_map = inout_map
            
    def forward(self,x):
        x = x.to(device)
        
        #pre-model
        x = self.pre(x)
        
        #projection layer
        x = x.view(x.size(0),-1)
        x = self.proj(x)
        
        if (isinstance(self.inout_map,nn.Sequential)):
            x = x.T
        #inout_layer
        x = self.inout_map(x)
        
        return x