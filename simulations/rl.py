import torch  
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np  

class PolicyNetwork(nn.Module):
    '''
    num_inputs: the length of the feature respresentation of the states
    '''
    def __init__(self, num_inputs, num_actions):
        super(PolicyNetwork, self).__init__()

        self.num_actions = num_actions
        self.linear1 = nn.Linear(num_inputs, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, num_actions)
#         self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        # x = F.softmax(self.linear2(x), dim=1)
        x = F.relu(self.linear2(x))
        x = F.softmax(self.linear3(x), dim=1)
        
        return x 
    
    def get_action(self, state):
        # assuming state is in shape [number of features]
        # print('state.shape before', state.shape)
        # reshape state to [1, number of features]
        state = state.unsqueeze(0)
        # print('state.shape after', state.shape)
        probs = self.forward(Variable(state))
        # print('probs.shape', probs.shape)
#         print()
        # print('np.squeeze(probs.detach().cpu().numpy()).shape', np.squeeze(probs.detach().cpu().numpy()).shape)
        # print('np.squeeze(probs.detach().cpu().numpy())', np.squeeze(probs.detach().cpu().clone().numpy()))
        # print()
        # print('type(probs)', type(probs))
        # print('probs.detach().squeeze().shape', probs.detach().squeeze().shape)
        # print('probs.detach().squeeze()', probs.detach().squeeze())
        # print(probs.detach().squeeze().sum())
        # print('probs', probs)
        # print()
        # probs = probs.to(torch.device("cuda"))
        highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().cpu().numpy()))
        # print(highest_prob_action)
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        # print(log_prob)
        # print()
        
        return highest_prob_action, log_prob